//==============================================================================
// throttle_enhanced.h - Enhanced Throttling System
//==============================================================================

#ifndef THROTTLE_ENHANCED_H
#define THROTTLE_ENHANCED_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct cpu_profile_t cpu_profile_t;

/**
 * @brief Throttling modes
 */
typedef enum {
    THROTTLE_MODE_SIMPLE,      // Original windowed refill
    THROTTLE_MODE_TOKEN_BUCKET // Time-based token bucket
} throttle_mode_t;

/**
 * @brief Prefetch priority levels
 */
typedef enum {
    PREFETCH_PRIO_CRITICAL = 0,
    PREFETCH_PRIO_HIGH = 1,
    PREFETCH_PRIO_MEDIUM = 2,
    PREFETCH_PRIO_LOW = 3
} prefetch_priority_t;

//==============================================================================
// INITIALIZATION & CLEANUP
//==============================================================================

/**
 * @brief Initialize enhanced throttling system
 * @param cpu_profile CPU architecture profile
 */
void init_throttling_enhanced(const cpu_profile_t *cpu_profile);

//==============================================================================
// CONFIGURATION
//==============================================================================

/**
 * @brief Set throttling mode
 * @param mode Throttling mode to use
 */
void set_throttle_mode(throttle_mode_t mode);

/**
 * @brief Configure token bucket parameters
 * @param l1_capacity Token capacity for L1 (T0/T1) prefetches
 * @param l2_capacity Token capacity for L2 (T2) prefetches
 * @param nta_capacity Token capacity for NTA (streaming) prefetches
 * @param refill_cycles Cycles between token refills
 * @param tokens_per_refill Tokens added per refill
 */
void configure_token_bucket(
    int l1_capacity,
    int l2_capacity,
    int nta_capacity,
    uint64_t refill_cycles,
    int tokens_per_refill
);

/**
 * @brief Enable/disable statistics collection
 * @param enable True to enable, false to disable
 */
void set_throttle_statistics(bool enable);

//==============================================================================
// PREFETCH OPERATIONS
//==============================================================================

/**
 * @brief Throttled prefetch with priority
 * @param addr Address to prefetch
 * @param hint Prefetch hint (_MM_HINT_T0, T1, T2, NTA)
 * @param priority Priority level
 * @param do_prefetch_fn Actual prefetch function to call if approved
 * @return True if prefetch was issued, false if throttled
 */
bool prefetch_throttled_enhanced(
    const void *addr,
    int hint,
    prefetch_priority_t priority,
    void (*do_prefetch_fn)(const void*, int)
);

//==============================================================================
// STATISTICS & MONITORING
//==============================================================================

/**
 * @brief Get throttling statistics
 * @param total_requested Total prefetch requests
 * @param total_issued Total prefetches actually issued
 * @param total_throttled Total prefetches throttled
 * @param critical_issued Critical priority prefetches issued
 * @param throttle_rate Throttle rate (0.0 to 1.0)
 */
void get_throttle_stats(
    uint64_t *total_requested,
    uint64_t *total_issued,
    uint64_t *total_throttled,
    uint64_t *critical_issued,
    double *throttle_rate
);

/**
 * @brief Print throttling statistics to stdout
 */
void print_throttle_stats(void);

//==============================================================================
// AUTO-TUNING
//==============================================================================

/**
 * @brief Auto-tune token bucket parameters based on observed behavior
 * 
 * Call periodically (e.g., every 1000 FFT calls) to adjust token bucket
 * capacities based on observed throttle rate. Aims to keep throttle rate
 * between 10-30%.
 */
void autotune_token_bucket(void);

#ifdef __cplusplus
}
#endif

#endif // THROTTLE_ENHANCED_H

