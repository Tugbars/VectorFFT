//==============================================================================
// ENHANCED THROTTLING MODEL - Token Bucket with Time Decay
// STANDALONE MODULE - Optional integration with prefetch_strategy.c
//
// Compilation modes:
//   1. Standalone: compile without HFFT_USE_ENHANCED_THROTTLE
//   2. Integrated: compile with -DHFFT_USE_ENHANCED_THROTTLE
//==============================================================================

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>  // for getenv, atoi

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
        int max_outstanding;
    } simple;
    
    // Token bucket mode
    struct {
        token_bucket_t l1_bucket;    // For T0/T1 hints
        token_bucket_t l2_bucket;    // For T2 hints
        token_bucket_t nta_bucket;   // For NTA hints
        int critical_bypass_count;   // Track bypassed criticals
    } token;
    
    // Statistics for comparison (per-thread)
    struct {
        uint64_t total_requested;
        uint64_t total_issued;
        uint64_t total_throttled;
        uint64_t critical_issued;
    } stats;
} prefetch_throttle_enhanced_t;

//==============================================================================
// GLOBAL STATE
//==============================================================================

// Global configuration
static struct {
    throttle_mode_t mode;
    bool enable_statistics;
    
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

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================

/**
 * @brief Helper to get environment variable as int with default
 */
static int getenv_int(const char* key, int def) {
    const char* v = getenv(key);
    return v ? atoi(v) : def;
}

/**
 * @brief Read CPU timestamp counter (inline for performance)
 * Portable: uses rdtsc on x86, clock_gettime elsewhere
 */
static inline uint64_t read_tsc_inline(void) {
#if HAS_X86_INTRINSICS
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
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
    bucket->refill_interval = refill_interval;
    bucket->tokens_per_refill = tokens_per_refill;
}

/**
 * @brief Apply global token config to live buckets
 * Call this after modifying g_throttle_config.token_config
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
    
    for (int i = 0; i < 3; ++i) {
        buckets[i]->capacity = capacities[i];
        buckets[i]->refill_interval = g_throttle_config.token_config.refill_cycles;
        buckets[i]->tokens_per_refill = g_throttle_config.token_config.tokens_per_refill;
        
        // Clamp current tokens to new capacity
        if (buckets[i]->tokens > buckets[i]->capacity) {
            buckets[i]->tokens = buckets[i]->capacity;
        }
    }
}

/**
 * @brief Refill tokens based on elapsed time (with overflow protection)
 */
static inline void refill_bucket(token_bucket_t *bucket) {
    uint64_t now = read_tsc_inline();
    uint64_t elapsed = now - bucket->last_refill;
    
    if (elapsed >= bucket->refill_interval) {
        // Calculate how many refill periods have passed (keep as uint64_t)
        uint64_t periods64 = elapsed / bucket->refill_interval;
        
        // Clamp to prevent overflow
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
        g_throttle_config.token_config.refill_cycles = lat ? lat / 2 : 1000;
    }
    
    // Apply environment variable overrides
    g_throttle_enhanced.simple.window_size = 
        getenv_int("HFFT_THROTTLE_WINDOW", 8);
    g_throttle_config.token_config.l1_capacity = 
        getenv_int("HFFT_TB_L1_CAP", g_throttle_config.token_config.l1_capacity);
    g_throttle_config.token_config.l2_capacity = 
        getenv_int("HFFT_TB_L2_CAP", g_throttle_config.token_config.l2_capacity);
    g_throttle_config.token_config.nta_capacity = 
        getenv_int("HFFT_TB_NTA_CAP", g_throttle_config.token_config.nta_capacity);
    g_throttle_config.token_config.refill_cycles = 
        getenv_int("HFFT_TB_REFILL", g_throttle_config.token_config.refill_cycles);
    
    // Initialize simple mode state
    g_throttle_enhanced.simple.max_outstanding = 
        cpu_profile ? cpu_profile->prefetch_buffers : 20;
    g_throttle_enhanced.simple.budget_remaining = 
        g_throttle_enhanced.simple.max_outstanding;
    g_throttle_enhanced.simple.window_counter = 0;
    g_throttle_enhanced.simple.issued_count = 0;
    
    // Initialize token bucket mode state
    const uint64_t refill_cycles = g_throttle_config.token_config.refill_cycles;
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
    
    // Critical priority always bypasses throttling
    if (priority == PREFETCH_PRIO_CRITICAL) {
        if (prefetch_impl) {
            prefetch_impl(addr, hint);
        }
        
        if (g_throttle_enhanced.mode == THROTTLE_MODE_TOKEN_BUCKET) {
            g_throttle_enhanced.token.critical_bypass_count++;
        }
        
        if (g_throttle_config.enable_statistics) {
            g_throttle_enhanced.stats.critical_issued++;
            g_throttle_enhanced.stats.total_issued++;
        }
        
        return true;
    }
    
    // Check throttling based on mode
    bool can_issue = false;
    
    if (g_throttle_enhanced.mode == THROTTLE_MODE_SIMPLE) {
        can_issue = can_prefetch_simple();
        if (can_issue && prefetch_impl) {
            prefetch_impl(addr, hint);
            record_prefetch_simple(true);
        }
    } else { // THROTTLE_MODE_TOKEN_BUCKET
        can_issue = can_prefetch_token_bucket(hint);
        if (can_issue && prefetch_impl) {
            prefetch_impl(addr, hint);
        }
    }
    
    if (g_throttle_config.enable_statistics) {
        if (can_issue) {
            g_throttle_enhanced.stats.total_issued++;
        } else {
            g_throttle_enhanced.stats.total_throttled++;
        }
    }
    
    return can_issue;
}

/**
 * @brief Set throttling mode
 * Takes effect immediately in current thread
 */
void set_throttle_mode(throttle_mode_t mode) {
    g_throttle_config.mode = mode;
    g_throttle_enhanced.mode = mode;  // Apply immediately to TLS
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
    g_throttle_config.token_config.refill_cycles = refill_cycles;
    g_throttle_config.token_config.tokens_per_refill = tokens_per_refill;
    
    // Apply changes to live buckets
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
 * @brief Print throttling statistics
 */
void print_throttle_stats(void) {
    // Always print mode for debugging, even if stats disabled
    printf("=== Throttling Statistics (this thread) ===\n");
    printf("Mode: %s\n", 
        g_throttle_enhanced.mode == THROTTLE_MODE_SIMPLE ? "Simple" : "Token Bucket");
    
    if (!g_throttle_config.enable_statistics) {
        printf("Statistics collection not enabled\n");
        printf("==========================================\n");
        return;
    }
    
    uint64_t req, issued, throttled, critical;
    double rate;
    get_throttle_stats(&req, &issued, &throttled, &critical, &rate);
    
    printf("Total requested: %llu\n", (unsigned long long)req);
    printf("Total issued: %llu\n", (unsigned long long)issued);
    printf("Total throttled: %llu\n", (unsigned long long)throttled);
    printf("Critical issued: %llu\n", (unsigned long long)critical);
    printf("Throttle rate: %.2f%%\n", rate * 100.0);
    
    if (g_throttle_enhanced.mode == THROTTLE_MODE_TOKEN_BUCKET) {
        printf("\nToken Bucket State:\n");
        printf("  L1 tokens: %d / %d\n", 
            g_throttle_enhanced.token.l1_bucket.tokens,
            g_throttle_enhanced.token.l1_bucket.capacity);
        printf("  L2 tokens: %d / %d\n",
            g_throttle_enhanced.token.l2_bucket.tokens,
            g_throttle_enhanced.token.l2_bucket.capacity);
        printf("  NTA tokens: %d / %d\n",
            g_throttle_enhanced.token.nta_bucket.tokens,
            g_throttle_enhanced.token.nta_bucket.capacity);
        printf("  Critical bypasses: %d\n",
            g_throttle_enhanced.token.critical_bypass_count);
    }
    printf("==========================================\n");
}

//==============================================================================
// AUTO-TUNING FOR TOKEN BUCKET PARAMETERS
//==============================================================================

/**
 * @brief Auto-tune token bucket based on observed behavior
 * 
 * Call this periodically (e.g., every 1000 FFT calls) to adjust parameters
 * Note: Not thread-safe. Call from single controller thread.
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
        // Under-throttling: reduce capacities slightly
        g_throttle_config.token_config.l1_capacity = 
            (g_throttle_config.token_config.l1_capacity * 9) / 10;
        g_throttle_config.token_config.l2_capacity = 
            (g_throttle_config.token_config.l2_capacity * 9) / 10;
        g_throttle_config.token_config.nta_capacity = 
            (g_throttle_config.token_config.nta_capacity * 9) / 10;
        
        // Clamp to minimums
        if (g_throttle_config.token_config.l1_capacity < 4)
            g_throttle_config.token_config.l1_capacity = 4;
        if (g_throttle_config.token_config.l2_capacity < 6)
            g_throttle_config.token_config.l2_capacity = 6;
        if (g_throttle_config.token_config.nta_capacity < 8)
            g_throttle_config.token_config.nta_capacity = 8;
            
    } else if (throttle_rate > 0.30) {
        // Over-throttling: increase capacities slightly
        g_throttle_config.token_config.l1_capacity = 
            (g_throttle_config.token_config.l1_capacity * 11) / 10;
        g_throttle_config.token_config.l2_capacity = 
            (g_throttle_config.token_config.l2_capacity * 11) / 10;
        g_throttle_config.token_config.nta_capacity = 
            (g_throttle_config.token_config.nta_capacity * 11) / 10;
        
        // Clamp to CPU limits (from profile)
        int max_buffers = g_throttle_config.prefetch_buffers_cap;
        if (g_throttle_config.token_config.l1_capacity > max_buffers)
            g_throttle_config.token_config.l1_capacity = max_buffers;
        if (g_throttle_config.token_config.l2_capacity > max_buffers)
            g_throttle_config.token_config.l2_capacity = max_buffers;
        if (g_throttle_config.token_config.nta_capacity > max_buffers)
            g_throttle_config.token_config.nta_capacity = max_buffers;
    }
    
    // Apply changes to live buckets
    apply_token_config_to_buckets();
    
    // Reset statistics for next tuning period
    g_throttle_enhanced.stats.total_requested = 0;
    g_throttle_enhanced.stats.total_issued = 0;
    g_throttle_enhanced.stats.total_throttled = 0;
    g_throttle_enhanced.stats.critical_issued = 0;
}
