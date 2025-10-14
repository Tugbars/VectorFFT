//==============================================================================
// ENHANCED THROTTLING MODEL - Token Bucket with Time Decay
// STANDALONE MODULE - Optional integration with prefetch_strategy.c
//
// PATCHED VERSION - All critical and architectural fixes applied
//
// Compilation modes:
//   1. Standalone: compile without HFFT_USE_ENHANCED_THROTTLE
//   2. Integrated: compile with -DHFFT_USE_ENHANCED_THROTTLE
//==============================================================================

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// Include x86 intrinsics only on x86 platforms
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define HAS_X86_INTRINSICS 1
#else
#define HAS_X86_INTRINSICS 0
// Define fallback hint constants
#define _MM_HINT_T0  1
#define _MM_HINT_T1  2
#define _MM_HINT_T2  3
#define _MM_HINT_NTA 0
#endif

//==============================================================================
// STANDALONE TYPE DEFINITIONS
// These are only used if NOT linking with prefetch_strategy.c
//==============================================================================

#ifndef HFFT_USE_ENHANCED_THROTTLE

/**
 * @brief Standalone CPU profile (subset needed for throttling)
 */
typedef struct {
    const char *name;
    int prefetch_buffers;
    int prefetch_latency;
} cpu_profile_t;

/**
 * @brief Standalone prefetch priority
 */
typedef enum {
    PREFETCH_PRIO_LOW = 3,
    PREFETCH_PRIO_MEDIUM = 2,
    PREFETCH_PRIO_HIGH = 1,
    PREFETCH_PRIO_CRITICAL = 0
} prefetch_priority_t;

#endif // !HFFT_USE_ENHANCED_THROTTLE

//==============================================================================
// CORE TYPES (always defined)
//==============================================================================

/**
 * @brief Throttling modes
 */
typedef enum {
    THROTTLE_MODE_SIMPLE,      // Original windowed refill
    THROTTLE_MODE_TOKEN_BUCKET // Time-based token bucket
} throttle_mode_t;

/**
 * @brief Token bucket level abstraction (portable)
 */
typedef enum { 
    TB_LVL_L1, 
    TB_LVL_L2, 
    TB_LVL_NTA 
} tb_level_t;

/**
 * @brief Per-level token bucket state
 */
typedef struct {
    int tokens;              // Current available tokens
    int capacity;            // Maximum tokens
    uint64_t last_refill;    // TSC timestamp of last refill
    uint64_t refill_interval; // Cycles between refills
    int tokens_per_refill;   // Tokens added per refill
} token_bucket_t;

/**
 * @brief Enhanced prefetch throttle state (per-thread)
 */
typedef struct {
    throttle_mode_t mode;
    
    // Simple mode (original)
    struct {
        int issued_count;
        int budget_remaining;
        int window_size;
        int window_counter;
        int max_outstanding;  // Budget per window, not actual in-flight count
    } simple;
    
    // Token bucket mode
    struct {
        token_bucket_t l1_bucket;    // For T0/T1 hints
        token_bucket_t l2_bucket;    // For T2 hints
        token_bucket_t nta_bucket;   // For NTA hints
        uint32_t critical_bypass_count;   // Track bypassed criticals
    } token;
    
    // Statistics for comparison (per-thread)
    struct {
        uint64_t total_requested;
        uint64_t total_issued;
        uint64_t total_throttled;
        uint64_t critical_issued;
    } stats;
    
    // Initialization tracking
    bool initialized;
} prefetch_throttle_enhanced_t;

//==============================================================================
// GLOBAL STATE
//==============================================================================

// Global configuration with version tracking for cross-thread updates
static struct {
    throttle_mode_t mode;
    bool enable_statistics;
    uint32_t config_version;  // Incremented on config changes
    
    // Token bucket parameters (tunable)
    struct {
        int l1_capacity;
        int l2_capacity;
        int nta_capacity;
        uint64_t refill_cycles;  // Refill every N cycles
        int tokens_per_refill;
    } token_config;
    
    int prefetch_buffers_cap;  // From CPU profile
} g_throttle_config = {
    .mode = THROTTLE_MODE_SIMPLE,
    .enable_statistics = false,
    .config_version = 0,
    .token_config = {
        .l1_capacity = 8,        // Conservative for L1-bound prefetches
        .l2_capacity = 12,       // More aggressive for L2
        .nta_capacity = 16,      // Most aggressive for streaming
        .refill_cycles = 1000,   // Default, will be tuned
        .tokens_per_refill = 4
    },
    .prefetch_buffers_cap = 20  // Default fallback
};

static __thread prefetch_throttle_enhanced_t g_throttle_enhanced = {0};
static __thread uint32_t g_local_config_version = 0;

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================

/**
 * @brief Clamp refill interval to prevent divide-by-zero
 */
static inline uint64_t clamp_refill(uint64_t v) { 
    return v ? v : 1; 
}

/**
 * @brief Helper to get environment variable as bounded int with validation
 */
static int getenv_int_bounded(const char* key, int def, int minv, int maxv) {
    const char* v = getenv(key);
    if (!v || !*v) return def;
    
    char* end = NULL;
    errno = 0;
    long long t = strtoll(v, &end, 10);
    
    // Check for conversion errors
    if (end == v || errno == ERANGE) return def;
    
    // Clamp to bounds
    if (t < minv) t = minv;
    if (t > maxv) t = maxv;
    
    return (int)t;
}

/**
 * @brief Read CPU timestamp counter with serialization (inline for performance)
 * Uses RDTSCP + CPUID fence to prevent reordering
 */
static inline uint64_t read_tsc_inline(void) {
#if HAS_X86_INTRINSICS
    unsigned lo, hi, aux;
    // RDTSCP waits for prior instructions and reads TSC
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux) ::);
    // CPUID serializes subsequent instructions
    __asm__ __volatile__("cpuid" ::: "%rax", "%rbx", "%rcx", "%rdx");
    return ((uint64_t)hi << 32) | lo;
#else
    // Fallback: use monotonic clock (nanoseconds as proxy for cycles)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

/**
 * @brief Convert hint to portable bucket level
 */
static inline tb_level_t level_for_hint(int hint) {
#if HAS_X86_INTRINSICS
    switch (hint) {
        case _MM_HINT_T0:
        case _MM_HINT_T1: return TB_LVL_L1;
        case _MM_HINT_T2: return TB_LVL_L2;
        case _MM_HINT_NTA:
        default:          return TB_LVL_NTA;
    }
#else
    (void)hint;
    // On non-x86, assume streaming workloads (could also use TB_LVL_L1 for conservative)
    return TB_LVL_NTA;
#endif
}

/**
 * @brief Determine which bucket to use based on hint
 */
static inline token_bucket_t* get_bucket_for_hint(int hint) {
    tb_level_t level = level_for_hint(hint);
    
    switch (level) {
        case TB_LVL_L1:  return &g_throttle_enhanced.token.l1_bucket;
        case TB_LVL_L2:  return &g_throttle_enhanced.token.l2_bucket;
        case TB_LVL_NTA:
        default:         return &g_throttle_enhanced.token.nta_bucket;
    }
}

/**
 * @brief Initialize a token bucket
 */
static inline void init_token_bucket(
    token_bucket_t *bucket,
    int capacity,
    uint64_t refill_interval,
    int tokens_per_refill
) {
    bucket->tokens = capacity;
    bucket->capacity = capacity;
    bucket->last_refill = read_tsc_inline();
    bucket->refill_interval = clamp_refill(refill_interval);
    bucket->tokens_per_refill = tokens_per_refill;
}

/**
 * @brief Ensure bucket parameters match current config (lazy sync)
 * Handles cross-thread configuration updates
 */
static inline void ensure_bucket_params(
    token_bucket_t* b, 
    int cap, 
    uint64_t refi, 
    int tpr
) {
    refi = clamp_refill(refi);
    
    if (b->capacity != cap || b->refill_interval != refi || b->tokens_per_refill != tpr) {
        b->capacity = cap;
        b->refill_interval = refi;
        b->tokens_per_refill = tpr;
        
        // Clamp current tokens to new capacity
        if (b->tokens > b->capacity) {
            b->tokens = b->capacity;
        }
    }
}

/**
 * @brief Apply global token config to live buckets (current thread only)
 * Call this after modifying g_throttle_config.token_config
 * 
 * NOTE: This only updates the CURRENT thread's buckets.
 * Other threads will lazy-sync via ensure_bucket_params() or on next init.
 */
static inline void apply_token_config_to_buckets(void) {
    token_bucket_t *buckets[] = {
        &g_throttle_enhanced.token.l1_bucket,
        &g_throttle_enhanced.token.l2_bucket,
        &g_throttle_enhanced.token.nta_bucket
    };
    const int capacities[] = {
        g_throttle_config.token_config.l1_capacity,
        g_throttle_config.token_config.l2_capacity,
        g_throttle_config.token_config.nta_capacity
    };
    
    const uint64_t refi = clamp_refill(g_throttle_config.token_config.refill_cycles);
    const int tpr = g_throttle_config.token_config.tokens_per_refill;
    
    for (int i = 0; i < 3; ++i) {
        ensure_bucket_params(buckets[i], capacities[i], refi, tpr);
    }
    
    g_local_config_version = g_throttle_config.config_version;
}

/**
 * @brief Refill tokens based on elapsed time (with overflow protection)
 * FIXED: Guards against divide-by-zero and overflow
 */
static inline void refill_bucket(token_bucket_t *bucket) {
    // Guard against divide-by-zero (paused or misconfigured)
    if (bucket->refill_interval == 0) return;
    
    uint64_t now = read_tsc_inline();
    uint64_t elapsed = now - bucket->last_refill;
    
    if (elapsed < bucket->refill_interval) return;
    
    // Calculate how many refill periods have passed (keep as uint64_t)
    uint64_t periods64 = elapsed / bucket->refill_interval;
    
    // Clamp to prevent overflow (2^20 periods is already huge)
    if (periods64 > (1ULL << 20)) {
        periods64 = (1ULL << 20);
    }
    
    uint64_t add = periods64 * (uint64_t)bucket->tokens_per_refill;
    
    // Saturating add
    if ((uint64_t)bucket->tokens + add > (uint64_t)bucket->capacity) {
        bucket->tokens = bucket->capacity;
    } else {
        bucket->tokens += (int)add;
    }
    
    // Reduce drift by advancing exactly multiples of interval
    bucket->last_refill += periods64 * bucket->refill_interval;
}

/**
 * @brief Try to consume a token from bucket
 */
static inline bool consume_token(token_bucket_t *bucket) {
    refill_bucket(bucket);
    
    if (bucket->tokens > 0) {
        bucket->tokens--;
        return true;
    }
    return false;
}

/**
 * @brief Ensure thread-local state is initialized with defaults
 * Protects against use-before-init bugs
 */
static inline void ensure_initialized_defaults(void) {
    if (g_throttle_enhanced.initialized) return;
    
    // Check if all buckets are uninitialized (capacity == 0)
    if (g_throttle_enhanced.token.l1_bucket.capacity == 0 &&
        g_throttle_enhanced.token.l2_bucket.capacity == 0 &&
        g_throttle_enhanced.token.nta_bucket.capacity == 0) {
        
        const uint64_t refi = clamp_refill(
            g_throttle_config.token_config.refill_cycles ?: 1000
        );
        const int tpr = g_throttle_config.token_config.tokens_per_refill ?: 4;
        
        init_token_bucket(&g_throttle_enhanced.token.l1_bucket, 8, refi, tpr);
        init_token_bucket(&g_throttle_enhanced.token.l2_bucket, 12, refi, tpr);
        init_token_bucket(&g_throttle_enhanced.token.nta_bucket, 16, refi, tpr);
        
        g_throttle_enhanced.mode = g_throttle_config.mode;
        g_throttle_enhanced.simple.max_outstanding = 20;
        g_throttle_enhanced.simple.window_size = 8;
        g_throttle_enhanced.initialized = true;
    }
}

/**
 * @brief Check if config has changed and sync if needed
 */
static inline void sync_config_if_changed(void) {
    if (g_local_config_version != g_throttle_config.config_version) {
        apply_token_config_to_buckets();
        g_throttle_enhanced.mode = g_throttle_config.mode;
    }
}

//==============================================================================
// STANDALONE PREFETCH IMPLEMENTATION (used when not integrated)
//==============================================================================

#ifndef HFFT_USE_ENHANCED_THROTTLE

/**
 * @brief Standalone prefetch dispatch
 */
static inline void do_prefetch_standalone(const void *addr, int hint) {
#if HAS_X86_INTRINSICS
    switch (hint) {
        case _MM_HINT_T0:  _mm_prefetch((const char*)addr, _MM_HINT_T0);  break;
        case _MM_HINT_T1:  _mm_prefetch((const char*)addr, _MM_HINT_T1);  break;
        case _MM_HINT_T2:  _mm_prefetch((const char*)addr, _MM_HINT_T2);  break;
        case _MM_HINT_NTA: _mm_prefetch((const char*)addr, _MM_HINT_NTA); break;
        default:           _mm_prefetch((const char*)addr, _MM_HINT_T0);  break;
    }
#else
    // locality 3≈T0, 2≈T1, 1≈T2, 0≈NTA (rough mapping)
    int locality = (hint==_MM_HINT_NTA) ? 0 : (hint==_MM_HINT_T2 ? 1 : (hint==_MM_HINT_T1 ? 2 : 3));
    __builtin_prefetch(addr, 0, locality);
#endif
}

#endif // !HFFT_USE_ENHANCED_THROTTLE

//==============================================================================
// PUBLIC API
//==============================================================================

/**
 * @brief Initialize enhanced throttling system
 */
void init_throttling_enhanced(const cpu_profile_t *cpu_profile) {
    g_throttle_enhanced.mode = g_throttle_config.mode;
    
    // Store CPU profile limits and apply env overrides
    if (cpu_profile) {
        g_throttle_config.prefetch_buffers_cap = cpu_profile->prefetch_buffers;
        
        // Set refill interval based on prefetch latency (~half round-trip)
        uint64_t lat = (uint64_t)cpu_profile->prefetch_latency;
        g_throttle_config.token_config.refill_cycles = clamp_refill(lat ? lat / 2 : 1000);
    }
    
    // Apply environment variable overrides with bounds checking
    g_throttle_enhanced.simple.window_size = 
        getenv_int_bounded("HFFT_THROTTLE_WINDOW", 8, 1, 1000);
    
    g_throttle_config.token_config.l1_capacity = 
        getenv_int_bounded("HFFT_TB_L1_CAP", 
            g_throttle_config.token_config.l1_capacity, 1, 64);
    
    g_throttle_config.token_config.l2_capacity = 
        getenv_int_bounded("HFFT_TB_L2_CAP", 
            g_throttle_config.token_config.l2_capacity, 1, 64);
    
    g_throttle_config.token_config.nta_capacity = 
        getenv_int_bounded("HFFT_TB_NTA_CAP", 
            g_throttle_config.token_config.nta_capacity, 1, 64);
    
    g_throttle_config.token_config.refill_cycles = 
        clamp_refill(getenv_int_bounded("HFFT_TB_REFILL", 
            g_throttle_config.token_config.refill_cycles, 1, 1000000));
    
    g_throttle_config.token_config.tokens_per_refill = 
        getenv_int_bounded("HFFT_TB_TOKENS_PER_REFILL", 
            g_throttle_config.token_config.tokens_per_refill, 1, 64);
    
    // Mode override
    int mode_env = getenv_int_bounded("HFFT_THROTTLE_MODE", -1, 0, 1);
    if (mode_env >= 0) {
        g_throttle_config.mode = (throttle_mode_t)mode_env;
        g_throttle_enhanced.mode = g_throttle_config.mode;
    }
    
    // Statistics override
    int stats_env = getenv_int_bounded("HFFT_THROTTLE_STATS", -1, 0, 1);
    if (stats_env >= 0) {
        g_throttle_config.enable_statistics = (bool)stats_env;
    }
    
    // Initialize simple mode state
    g_throttle_enhanced.simple.max_outstanding = 
        cpu_profile ? cpu_profile->prefetch_buffers : 20;
    g_throttle_enhanced.simple.budget_remaining = 
        g_throttle_enhanced.simple.max_outstanding;
    g_throttle_enhanced.simple.window_counter = 0;
    g_throttle_enhanced.simple.issued_count = 0;
    
    // Initialize token bucket mode state
    const uint64_t refill_cycles = clamp_refill(g_throttle_config.token_config.refill_cycles);
    const int tokens_per = g_throttle_config.token_config.tokens_per_refill;
    
    init_token_bucket(
        &g_throttle_enhanced.token.l1_bucket,
        g_throttle_config.token_config.l1_capacity,
        refill_cycles,
        tokens_per
    );
    
    init_token_bucket(
        &g_throttle_enhanced.token.l2_bucket,
        g_throttle_config.token_config.l2_capacity,
        refill_cycles,
        tokens_per
    );
    
    init_token_bucket(
        &g_throttle_enhanced.token.nta_bucket,
        g_throttle_config.token_config.nta_capacity,
        refill_cycles,
        tokens_per
    );
    
    g_throttle_enhanced.token.critical_bypass_count = 0;
    
    // Reset statistics
    if (g_throttle_config.enable_statistics) {
        g_throttle_enhanced.stats.total_requested = 0;
        g_throttle_enhanced.stats.total_issued = 0;
        g_throttle_enhanced.stats.total_throttled = 0;
        g_throttle_enhanced.stats.critical_issued = 0;
    }
    
    g_throttle_enhanced.initialized = true;
    g_local_config_version = g_throttle_config.config_version;
}

/**
 * @brief Check if prefetch can be issued (simple mode)
 */
static inline bool can_prefetch_simple(void) {
    g_throttle_enhanced.simple.window_counter++;
    
    if (g_throttle_enhanced.simple.window_counter >= g_throttle_enhanced.simple.window_size) {
        // Window reset
        g_throttle_enhanced.simple.window_counter = 0;
        g_throttle_enhanced.simple.budget_remaining = g_throttle_enhanced.simple.max_outstanding;
        g_throttle_enhanced.simple.issued_count = 0;
    }
    
    return (g_throttle_enhanced.simple.budget_remaining > 0);
}

/**
 * @brief Record prefetch issued (simple mode)
 */
static inline void record_prefetch_simple(bool counts_against_budget) {
    if (counts_against_budget && g_throttle_enhanced.simple.budget_remaining > 0) {
        g_throttle_enhanced.simple.budget_remaining--;
    }
    g_throttle_enhanced.simple.issued_count++;
}

/**
 * @brief Check if prefetch can be issued (token bucket mode)
 */
static inline bool can_prefetch_token_bucket(int hint) {
    token_bucket_t *bucket = get_bucket_for_hint(hint);
    return consume_token(bucket);
}

/**
 * @brief Main throttled prefetch function
 * 
 * @param addr Address to prefetch
 * @param hint Prefetch hint (T0/T1/T2/NTA)
 * @param priority Prefetch priority
 * @param do_prefetch_fn Function pointer to actual prefetch implementation
 *                       (pass NULL to use built-in standalone version)
 * 
 * When integrated with prefetch_strategy.c:
 *   - Pass the do_prefetch() function from that module
 * When standalone:
 *   - Pass NULL to use the built-in implementation
 */
bool prefetch_throttled_enhanced(
    const void *addr,
    int hint,
    prefetch_priority_t priority,
    void (*do_prefetch_fn)(const void*, int)
) {
    // Ensure initialized (guards against use-before-init)
    ensure_initialized_defaults();
    
    // Sync config if changed (handles cross-thread updates)
    sync_config_if_changed();
    
    // Ensure bucket params are current
    if (g_throttle_enhanced.mode == THROTTLE_MODE_TOKEN_BUCKET) {
        const uint64_t refi = clamp_refill(g_throttle_config.token_config.refill_cycles);
        const int tpr = g_throttle_config.token_config.tokens_per_refill;
        
        ensure_bucket_params(&g_throttle_enhanced.token.l1_bucket, 
            g_throttle_config.token_config.l1_capacity, refi, tpr);
        ensure_bucket_params(&g_throttle_enhanced.token.l2_bucket, 
            g_throttle_config.token_config.l2_capacity, refi, tpr);
        ensure_bucket_params(&g_throttle_enhanced.token.nta_bucket, 
            g_throttle_config.token_config.nta_capacity, refi, tpr);
    }
    
    if (g_throttle_config.enable_statistics) {
        g_throttle_enhanced.stats.total_requested++;
    }
    
    // Select prefetch implementation
    void (*prefetch_impl)(const void*, int) = do_prefetch_fn;
#ifndef HFFT_USE_ENHANCED_THROTTLE
    if (!prefetch_impl) {
        prefetch_impl = do_prefetch_standalone;
    }
#endif
    
    // Track whether we actually dispatched (fixes ghost dispatch bug)
    bool dispatched = false;
    
    // Critical priority always bypasses throttling
    if (priority == PREFETCH_PRIO_CRITICAL) {
        if (prefetch_impl) {
            prefetch_impl(addr, hint);
            dispatched = true;
        }
        
        if (g_throttle_enhanced.mode == THROTTLE_MODE_TOKEN_BUCKET) {
            g_throttle_enhanced.token.critical_bypass_count++;
        }
        
        if (g_throttle_config.enable_statistics) {
            if (dispatched) {
                g_throttle_enhanced.stats.total_issued++;
            }
            g_throttle_enhanced.stats.critical_issued++;
        }
        
        return dispatched;
    }
    
    // Check throttling based on mode
    bool can_issue = false;
    
    if (g_throttle_enhanced.mode == THROTTLE_MODE_SIMPLE) {
        can_issue = can_prefetch_simple();
        if (can_issue && prefetch_impl) {
            prefetch_impl(addr, hint);
            dispatched = true;
            record_prefetch_simple(true);
        }
    } else { // THROTTLE_MODE_TOKEN_BUCKET
        can_issue = can_prefetch_token_bucket(hint);
        if (can_issue && prefetch_impl) {
            prefetch_impl(addr, hint);
            dispatched = true;
        }
    }
    
    if (g_throttle_config.enable_statistics) {
        if (dispatched) {
            g_throttle_enhanced.stats.total_issued++;
        } else {
            g_throttle_enhanced.stats.total_throttled++;
        }
    }
    
    return dispatched;
}

/**
 * @brief Set throttling mode
 * Takes effect immediately in current thread
 */
void set_throttle_mode(throttle_mode_t mode) {
    g_throttle_config.mode = mode;
    g_throttle_enhanced.mode = mode;
    g_throttle_config.config_version++;
}

/**
 * @brief Configure token bucket parameters
 */
void configure_token_bucket(
    int l1_capacity,
    int l2_capacity,
    int nta_capacity,
    uint64_t refill_cycles,
    int tokens_per_refill
) {
    g_throttle_config.token_config.l1_capacity = l1_capacity;
    g_throttle_config.token_config.l2_capacity = l2_capacity;
    g_throttle_config.token_config.nta_capacity = nta_capacity;
    g_throttle_config.token_config.refill_cycles = clamp_refill(refill_cycles);
    g_throttle_config.token_config.tokens_per_refill = tokens_per_refill;
    
    g_throttle_config.config_version++;
    
    // Apply changes to current thread's live buckets
    apply_token_config_to_buckets();
}

/**
 * @brief Enable/disable statistics collection
 */
void set_throttle_statistics(bool enable) {
    g_throttle_config.enable_statistics = enable;
}

/**
 * @brief Get throttling statistics (with null-safe out params)
 */
void get_throttle_stats(
    uint64_t *total_requested,
    uint64_t *total_issued,
    uint64_t *total_throttled,
    uint64_t *critical_issued,
    double *throttle_rate
) {
    if (!g_throttle_config.enable_statistics) {
        if (total_requested) *total_requested = 0;
        if (total_issued) *total_issued = 0;
        if (total_throttled) *total_throttled = 0;
        if (critical_issued) *critical_issued = 0;
        if (throttle_rate) *throttle_rate = 0.0;
        return;
    }
    
    if (total_requested) *total_requested = g_throttle_enhanced.stats.total_requested;
    if (total_issued) *total_issued = g_throttle_enhanced.stats.total_issued;
    if (total_throttled) *total_throttled = g_throttle_enhanced.stats.total_throttled;
    if (critical_issued) *critical_issued = g_throttle_enhanced.stats.critical_issued;
    
    if (throttle_rate) {
        if (g_throttle_enhanced.stats.total_requested > 0) {
            *throttle_rate = (double)g_throttle_enhanced.stats.total_throttled / 
                            (double)g_throttle_enhanced.stats.total_requested;
        } else {
            *throttle_rate = 0.0;
        }
    }
}

/**
 * @brief Print throttling statistics (thread-safe via local buffer)
 */
void print_throttle_stats(void) {
    // Buffer output to avoid interleaved prints in multi-threaded context
    char buffer[2048];
    int pos = 0;
    
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
        "=== Throttling Statistics (this thread) ===\n");
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
        "Mode: %s\n", 
        g_throttle_enhanced.mode == THROTTLE_MODE_SIMPLE ? "Simple" : "Token Bucket");
    
    if (!g_throttle_config.enable_statistics) {
        pos += snprintf(buffer + pos, sizeof(buffer) - pos,
            "Statistics collection not enabled\n");
        pos += snprintf(buffer + pos, sizeof(buffer) - pos,
            "==========================================\n");
        printf("%s", buffer);
        return;
    }
    
    uint64_t req, issued, throttled, critical;
    double rate;
    get_throttle_stats(&req, &issued, &throttled, &critical, &rate);
    
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
        "Total requested: %llu\n", (unsigned long long)req);
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
        "Total issued: %llu\n", (unsigned long long)issued);
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
        "Total throttled: %llu\n", (unsigned long long)throttled);
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
        "Critical issued: %llu\n", (unsigned long long)critical);
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
        "Throttle rate: %.2f%%\n", rate * 100.0);
    
    if (g_throttle_enhanced.mode == THROTTLE_MODE_TOKEN_BUCKET) {
        pos += snprintf(buffer + pos, sizeof(buffer) - pos,
            "\nToken Bucket State:\n");
        pos += snprintf(buffer + pos, sizeof(buffer) - pos,
            "  L1 tokens: %d / %d\n", 
            g_throttle_enhanced.token.l1_bucket.tokens,
            g_throttle_enhanced.token.l1_bucket.capacity);
        pos += snprintf(buffer + pos, sizeof(buffer) - pos,
            "  L2 tokens: %d / %d\n",
            g_throttle_enhanced.token.l2_bucket.tokens,
            g_throttle_enhanced.token.l2_bucket.capacity);
        pos += snprintf(buffer + pos, sizeof(buffer) - pos,
            "  NTA tokens: %d / %d\n",
            g_throttle_enhanced.token.nta_bucket.tokens,
            g_throttle_enhanced.token.nta_bucket.capacity);
        pos += snprintf(buffer + pos, sizeof(buffer) - pos,
            "  Critical bypasses: %u\n",
            g_throttle_enhanced.token.critical_bypass_count);
    }
    pos += snprintf(buffer + pos, sizeof(buffer) - pos,
        "==========================================\n");
    
    printf("%s", buffer);
}

//==============================================================================
// AUTO-TUNING FOR TOKEN BUCKET PARAMETERS
//==============================================================================

/**
 * @brief Auto-tune token bucket based on observed behavior
 * 
 * Call this periodically (e.g., every 1000 FFT calls) to adjust parameters
 * Note: Not thread-safe. Call from single controller thread.
 * 
 * IMPROVED: Now adjusts both rate (tokens_per_refill) and capacity
 */
void autotune_token_bucket(void) {
    if (g_throttle_enhanced.mode != THROTTLE_MODE_TOKEN_BUCKET) return;
    if (!g_throttle_config.enable_statistics) return;
    
    double throttle_rate;
    uint64_t req, issued, throttled, critical;
    get_throttle_stats(&req, &issued, &throttled, &critical, &throttle_rate);
    
    if (req < 1000) return; // Need sufficient samples
    
    // Goal: keep throttle rate between 10-30%
    // Too low = wasting MSHR, too high = starving performance
    
    if (throttle_rate < 0.10) {
        // Under-throttling: reduce rate first (smoother), then capacity
        if (g_throttle_config.token_config.tokens_per_refill > 1) {
            g_throttle_config.token_config.tokens_per_refill = 
                (g_throttle_config.token_config.tokens_per_refill * 9) / 10;
        } else {
            // Rate already minimal, reduce capacities
            g_throttle_config.token_config.l1_capacity = 
                (g_throttle_config.token_config.l1_capacity * 9) / 10;
            g_throttle_config.token_config.l2_capacity = 
                (g_throttle_config.token_config.l2_capacity * 9) / 10;
            g_throttle_config.token_config.nta_capacity = 
                (g_throttle_config.token_config.nta_capacity * 9) / 10;
        }
        
        // Clamp to minimums
        if (g_throttle_config.token_config.tokens_per_refill < 1)
            g_throttle_config.token_config.tokens_per_refill = 1;
        if (g_throttle_config.token_config.l1_capacity < 4)
            g_throttle_config.token_config.l1_capacity = 4;
        if (g_throttle_config.token_config.l2_capacity < 6)
            g_throttle_config.token_config.l2_capacity = 6;
        if (g_throttle_config.token_config.nta_capacity < 8)
            g_throttle_config.token_config.nta_capacity = 8;
            
    } else if (throttle_rate > 0.30) {
        // Over-throttling: increase rate first (smoother), then capacity
        if (g_throttle_config.token_config.tokens_per_refill < 64) {
            g_throttle_config.token_config.tokens_per_refill++;
        } else {
            // Rate already maxed, increase capacities
            g_throttle_config.token_config.l1_capacity = 
                (g_throttle_config.token_config.l1_capacity * 11) / 10;
            g_throttle_config.token_config.l2_capacity = 
                (g_throttle_config.token_config.l2_capacity * 11) / 10;
            g_throttle_config.token_config.nta_capacity = 
                (g_throttle_config.token_config.nta_capacity * 11) / 10;
        }
        
        // Clamp to CPU limits (from profile)
        int max_buffers = g_throttle_config.prefetch_buffers_cap;
        if (g_throttle_config.token_config.l1_capacity > max_buffers)
            g_throttle_config.token_config.l1_capacity = max_buffers;
        if (g_throttle_config.token_config.l2_capacity > max_buffers)
            g_throttle_config.token_config.l2_capacity = max_buffers;
        if (g_throttle_config.token_config.nta_capacity > max_buffers)
            g_throttle_config.token_config.nta_capacity = max_buffers;
    }
    
    // Increment config version to propagate changes
    g_throttle_config.config_version++;
    
    // Apply changes to live buckets (current thread)
    apply_token_config_to_buckets();
    
    // Reset statistics for next tuning period
    g_throttle_enhanced.stats.total_requested = 0;
    g_throttle_enhanced.stats.total_issued = 0;
    g_throttle_enhanced.stats.total_throttled = 0;
    g_throttle_enhanced.stats.critical_issued = 0;
}
