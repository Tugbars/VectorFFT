//==============================================================================
// ENHANCED THROTTLING MODEL - Token Bucket with Time Decay
//==============================================================================

#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Throttling modes
 */
typedef enum {
    THROTTLE_MODE_SIMPLE,      // Original windowed refill
    THROTTLE_MODE_TOKEN_BUCKET // Time-based token bucket
} throttle_mode_t;

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
    
    // Statistics for comparison
    struct {
        uint64_t total_requested;
        uint64_t total_issued;
        uint64_t total_throttled;
        uint64_t critical_issued;
    } stats;
} prefetch_throttle_enhanced_t;

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
} g_throttle_config = {
    .mode = THROTTLE_MODE_SIMPLE,
    .enable_statistics = false,
    .token_config = {
        .l1_capacity = 8,        // Conservative for L1-bound prefetches
        .l2_capacity = 12,       // More aggressive for L2
        .nta_capacity = 16,      // Most aggressive for streaming
        .refill_cycles = 1000,   // Refill ~every 1000 cycles (~300ns @ 3GHz)
        .tokens_per_refill = 4
    }
};

static __thread prefetch_throttle_enhanced_t g_throttle_enhanced = {0};

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================

/**
 * @brief Read CPU timestamp counter (inline for performance)
 */
static inline uint64_t read_tsc_inline(void) {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#else
    return 0; // Fallback: disable time-based throttling on non-x86
#endif
}

/**
 * @brief Determine which bucket to use based on hint
 */
static inline token_bucket_t* get_bucket_for_hint(int hint) {
    switch (hint) {
        case _MM_HINT_T0:
        case _MM_HINT_T1:
            return &g_throttle_enhanced.token.l1_bucket;
        case _MM_HINT_T2:
            return &g_throttle_enhanced.token.l2_bucket;
        case _MM_HINT_NTA:
        default:
            return &g_throttle_enhanced.token.nta_bucket;
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
 * @brief Refill tokens based on elapsed time
 */
static inline void refill_bucket(token_bucket_t *bucket) {
    uint64_t now = read_tsc_inline();
    uint64_t elapsed = now - bucket->last_refill;
    
    if (elapsed >= bucket->refill_interval) {
        // Calculate how many refill periods have passed
        int periods = (int)(elapsed / bucket->refill_interval);
        int new_tokens = periods * bucket->tokens_per_refill;
        
        // Add tokens, capped at capacity
        bucket->tokens += new_tokens;
        if (bucket->tokens > bucket->capacity) {
            bucket->tokens = bucket->capacity;
        }
        
        // Update last refill time (don't accumulate drift)
        bucket->last_refill = now;
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
// PUBLIC API
//==============================================================================

/**
 * @brief Initialize enhanced throttling system
 */
void init_throttling_enhanced(const cpu_profile_t *cpu_profile) {
    g_throttle_enhanced.mode = g_throttle_config.mode;
    
    // Initialize simple mode state
    g_throttle_enhanced.simple.max_outstanding = cpu_profile->prefetch_buffers;
    g_throttle_enhanced.simple.budget_remaining = g_throttle_enhanced.simple.max_outstanding;
    g_throttle_enhanced.simple.window_size = 8;
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
    
    // Critical priority always bypasses throttling
    if (priority == PREFETCH_PRIO_CRITICAL) {
        do_prefetch_fn(addr, hint);
        
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
        if (can_issue) {
            do_prefetch_fn(addr, hint);
            record_prefetch_simple(true);
        }
    } else { // THROTTLE_MODE_TOKEN_BUCKET
        can_issue = can_prefetch_token_bucket(hint);
        if (can_issue) {
            do_prefetch_fn(addr, hint);
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
 */
void set_throttle_mode(throttle_mode_t mode) {
    g_throttle_config.mode = mode;
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
}

/**
 * @brief Enable/disable statistics collection
 */
void set_throttle_statistics(bool enable) {
    g_throttle_config.enable_statistics = enable;
}

/**
 * @brief Get throttling statistics
 */
void get_throttle_stats(
    uint64_t *total_requested,
    uint64_t *total_issued,
    uint64_t *total_throttled,
    uint64_t *critical_issued,
    double *throttle_rate
) {
    if (!g_throttle_config.enable_statistics) {
        *total_requested = 0;
        *total_issued = 0;
        *total_throttled = 0;
        *critical_issued = 0;
        *throttle_rate = 0.0;
        return;
    }
    
    *total_requested = g_throttle_enhanced.stats.total_requested;
    *total_issued = g_throttle_enhanced.stats.total_issued;
    *total_throttled = g_throttle_enhanced.stats.total_throttled;
    *critical_issued = g_throttle_enhanced.stats.critical_issued;
    
    if (g_throttle_enhanced.stats.total_requested > 0) {
        *throttle_rate = (double)g_throttle_enhanced.stats.total_throttled / 
                        (double)g_throttle_enhanced.stats.total_requested;
    } else {
        *throttle_rate = 0.0;
    }
}

/**
 * @brief Print throttling statistics
 */
void print_throttle_stats(void) {
    if (!g_throttle_config.enable_statistics) {
        printf("Throttling statistics not enabled\n");
        return;
    }
    
    uint64_t req, issued, throttled, critical;
    double rate;
    get_throttle_stats(&req, &issued, &throttled, &critical, &rate);
    
    printf("=== Throttling Statistics ===\n");
    printf("Mode: %s\n", 
        g_throttle_enhanced.mode == THROTTLE_MODE_SIMPLE ? "Simple" : "Token Bucket");
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
    printf("=============================\n");
}

//==============================================================================
// AUTO-TUNING FOR TOKEN BUCKET PARAMETERS
//==============================================================================

/**
 * @brief Auto-tune token bucket based on observed behavior
 * 
 * Call this periodically (e.g., every 1000 FFT calls) to adjust parameters
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
        int max_buffers = 20; // Could get from g_cpu_profile
        if (g_throttle_config.token_config.l1_capacity > max_buffers)
            g_throttle_config.token_config.l1_capacity = max_buffers;
        if (g_throttle_config.token_config.l2_capacity > max_buffers)
            g_throttle_config.token_config.l2_capacity = max_buffers;
        if (g_throttle_config.token_config.nta_capacity > max_buffers)
            g_throttle_config.token_config.nta_capacity = max_buffers;
    }
    
    // Reset statistics for next tuning period
    g_throttle_enhanced.stats.total_requested = 0;
    g_throttle_enhanced.stats.total_issued = 0;
    g_throttle_enhanced.stats.total_throttled = 0;
    g_throttle_enhanced.stats.critical_issued = 0;
}
