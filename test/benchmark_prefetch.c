//==============================================================================
// TESTING & VALIDATION FRAMEWORK
// Comprehensive test suite for throttling and tuning systems
//==============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "throttle_enhanced.h"
#include "adaptive_tuning.h"

//==============================================================================
// TEST UTILITIES
//==============================================================================

#define TEST_PASSED "\033[32m✓ PASSED\033[0m"
#define TEST_FAILED "\033[31m✗ FAILED\033[0m"

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define ASSERT_TRUE(cond, msg) do { \
    g_tests_run++; \
    if (cond) { \
        g_tests_passed++; \
        printf("  %s: %s\n", TEST_PASSED, msg); \
    } else { \
        printf("  %s: %s\n", TEST_FAILED, msg); \
    } \
} while(0)

#define ASSERT_EQ(a, b, msg) ASSERT_TRUE((a) == (b), msg)
#define ASSERT_NEAR(a, b, eps, msg) ASSERT_TRUE(fabs((a) - (b)) < (eps), msg)

//==============================================================================
// MOCK CPU PROFILE FOR TESTING
//==============================================================================

static const cpu_profile_t test_cpu_profile = {
    .name = "Test CPU",
    .prefetch_buffers = 12,
    .prefetch_latency = 200,
    .l1_latency = 4,
    .l2_latency = 12,
    .l3_latency = 40,
    .has_write_prefetch = true,
    .has_strong_hwpf = false,
    .optimal_distance = {4, 6, 8, 12, 16, 20, 24, 32}
};

//==============================================================================
// TEST SUITE 1: THROTTLING MECHANISM
//==============================================================================

void test_simple_throttle_basic(void) {
    printf("\n=== Test: Simple Throttle Basic ===\n");
    
    set_throttle_mode(THROTTLE_MODE_SIMPLE);
    set_throttle_statistics(true);
    init_throttling_enhanced(&test_cpu_profile);
    
    // Issue prefetches up to budget
    int issued = 0;
    for (int i = 0; i < 20; i++) {
        bool result = prefetch_throttled_enhanced(
            (void*)(uintptr_t)i,
            _MM_HINT_T0,
            PREFETCH_PRIO_HIGH,
            do_prefetch
        );
        if (result) issued++;
    }
    
    uint64_t req, iss, throttled, critical;
    double rate;
    get_throttle_stats(&req, &iss, &throttled, &critical, &rate);
    
    ASSERT_EQ(req, 20, "Total requests correct");
    ASSERT_TRUE(issued < 20, "Some prefetches throttled");
    ASSERT_TRUE(issued == iss, "Issued count matches");
    
    printf("  Issued: %d/%d (throttle rate: %.1f%%)\n", issued, 20, rate * 100.0);
}

void test_critical_priority_bypass(void) {
    printf("\n=== Test: Critical Priority Bypass ===\n");
    
    set_throttle_mode(THROTTLE_MODE_SIMPLE);
    set_throttle_statistics(true);
    init_throttling_enhanced(&test_cpu_profile);
    
    // Fill up the budget with normal priority
    for (int i = 0; i < 20; i++) {
        prefetch_throttled_enhanced(
            (void*)(uintptr_t)i,
            _MM_HINT_T0,
            PREFETCH_PRIO_HIGH,
            do_prefetch
        );
    }
    
    // Now issue critical - should always succeed
    int critical_issued = 0;
    for (int i = 0; i < 5; i++) {
        bool result = prefetch_throttled_enhanced(
            (void*)(uintptr_t)(i + 1000),
            _MM_HINT_T0,
            PREFETCH_PRIO_CRITICAL,
            do_prefetch
        );
        if (result) critical_issued++;
    }
    
    ASSERT_EQ(critical_issued, 5, "All critical prefetches issued");
    
    uint64_t req, iss, throttled, critical;
    double rate;
    get_throttle_stats(&req, &iss, &throttled, &critical, &rate);
    
    ASSERT_EQ(critical, 5, "Critical count correct");
    printf("  Critical issued: %d/%d\n", critical_issued, 5);
}

void test_token_bucket_refill(void) {
    printf("\n=== Test: Token Bucket Refill ===\n");
    
    set_throttle_mode(THROTTLE_MODE_TOKEN_BUCKET);
    configure_token_bucket(4, 4, 4, 100, 2); // Small capacity, fast refill
    set_throttle_statistics(true);
    init_throttling_enhanced(&test_cpu_profile);
    
    // Exhaust tokens
    for (int i = 0; i < 10; i++) {
        prefetch_throttled_enhanced(
            (void*)(uintptr_t)i,
            _MM_HINT_T0,
            PREFETCH_PRIO_HIGH,
            do_prefetch
        );
    }
    
    uint64_t req1, iss1, thr1, crit1;
    double rate1;
    get_throttle_stats(&req1, &iss1, &thr1, &crit1, &rate1);
    
    // Simulate time passing (in real code, cycles would pass)
    // Here we just wait and issue more prefetches
    for (volatile int spin = 0; spin < 10000000; spin++);
    
    // Try again after refill
    int issued_after = 0;
    for (int i = 0; i < 10; i++) {
        bool result = prefetch_throttled_enhanced(
            (void*)(uintptr_t)(i + 100),
            _MM_HINT_T0,
            PREFETCH_PRIO_HIGH,
            do_prefetch
        );
        if (result) issued_after++;
    }
    
    ASSERT_TRUE(issued_after > 0, "Tokens refilled, some prefetches issued");
    printf("  Initial issued: %llu, After refill: %d\n", iss1, issued_after);
}

void test_per_level_token_buckets(void) {
    printf("\n=== Test: Per-Level Token Buckets ===\n");
    
    set_throttle_mode(THROTTLE_MODE_TOKEN_BUCKET);
    configure_token_bucket(
        4,   // L1: 4 tokens
        8,   // L2: 8 tokens
        12,  // NTA: 12 tokens
        1000, 2
    );
    set_throttle_statistics(true);
    init_throttling_enhanced(&test_cpu_profile);
    
    // Issue prefetches with different hints
    int t0_issued = 0, t2_issued = 0, nta_issued = 0;
    
    for (int i = 0; i < 20; i++) {
        if (prefetch_throttled_enhanced(
            (void*)(uintptr_t)i, _MM_HINT_T0, PREFETCH_PRIO_HIGH, do_prefetch))
            t0_issued++;
        
        if (prefetch_throttled_enhanced(
            (void*)(uintptr_t)(i+100), _MM_HINT_T2, PREFETCH_PRIO_HIGH, do_prefetch))
            t2_issued++;
        
        if (prefetch_throttled_enhanced(
            (void*)(uintptr_t)(i+200), _MM_HINT_NTA, PREFETCH_PRIO_HIGH, do_prefetch))
            nta_issued++;
    }
    
    // NTA should issue more than T2, T2 more than T0 (bigger capacity)
    ASSERT_TRUE(nta_issued > t2_issued, "NTA issued more than T2");
    ASSERT_TRUE(t2_issued > t0_issued, "T2 issued more than T0");
    
    printf("  T0: %d, T2: %d, NTA: %d\n", t0_issued, t2_issued, nta_issued);
}

//==============================================================================
// TEST SUITE 2: ADAPTIVE TUNING
//==============================================================================

void test_ewma_filter(void) {
    printf("\n=== Test: EWMA Filter ===\n");
    
    ewma_filter_t filter;
    ewma_init(&filter, 0.2);
    
    ASSERT_TRUE(!filter.initialized, "Filter starts uninitialized");
    
    // Feed samples
    double samples[] = {100.0, 102.0, 98.0, 101.0, 99.0};
    for (int i = 0; i < 5; i++) {
        ewma_update(&filter, samples[i]);
    }
    
    ASSERT_TRUE(filter.initialized, "Filter initialized after samples");
    ASSERT_TRUE(ewma_is_warmed_up(&filter, 5), "Filter warmed up");
    
    double value = ewma_get(&filter);
    ASSERT_TRUE(value > 95.0 && value < 105.0, "EWMA value in reasonable range");
    
    // EWMA should be smoother than raw samples
    double variance = 0.0;
    double mean = 100.0;
    for (int i = 0; i < 5; i++) {
        variance += (samples[i] - mean) * (samples[i] - mean);
    }
    variance /= 5.0;
    
    printf("  Sample variance: %.2f, EWMA value: %.2f\n", variance, value);
}

void test_simple_tuning_convergence(void) {
    printf("\n=== Test: Simple Tuning Convergence ===\n");
    
    set_tuning_mode(TUNING_MODE_SIMPLE);
    configure_tuning(0.2, 5, 0.02, 15, 4);
    
    int stages = 1;
    int dist = 8;
    int ws = 64 * 1024;
    int radix = 4;
    
    init_adaptive_tuning(stages, &dist, &ws, &radix);
    
    // Simulate FFT executions with improving performance
    for (int i = 0; i < 100; i++) {
        uint64_t start = profile_fft_start();
        
        // Simulate work (shorter time = better)
        int current_dist = get_tuned_distance(0);
        uint64_t work_cycles = 10000 + abs(current_dist - 12) * 500; // Optimal at 12
        
        // Simulate cycles passing
        volatile uint64_t dummy = 0;
        for (uint64_t j = 0; j < work_cycles; j++) dummy++;
        
        profile_fft_end(start, 1024, 0);
    }
    
    int final_dist = get_tuned_distance(0);
    
    // Should converge near optimal (12)
    ASSERT_TRUE(abs(final_dist - 12) <= 4, "Converged near optimal distance");
    
    printf("  Initial: %d, Final: %d (optimal: 12)\n", dist, final_dist);
    
    cleanup_adaptive_tuning();
}

void test_per_stage_independence(void) {
    printf("\n=== Test: Per-Stage Independence ===\n");
    
    set_tuning_mode(TUNING_MODE_PER_STAGE);
    configure_tuning(0.2, 5, 0.02, 20, 4);
    set_tuning_logging(false); // Suppress output for test
    
    int stages = 3;
    int dists[] = {4, 8, 16};
    int ws[] = {16*1024, 64*1024, 256*1024};
    int radixes[] = {4, 8, 16};
    
    init_adaptive_tuning(stages, dists, ws, radixes);
    
    // Simulate different optimal distances for each stage
    int optimal[] = {6, 12, 20};
    
    for (int iter = 0; iter < 100; iter++) {
        for (int stage = 0; stage < stages; stage++) {
            uint64_t start = profile_fft_start();
            
            int current = get_tuned_distance(stage);
            uint64_t cycles = 10000 + abs(current - optimal[stage]) * 500;
            
            volatile uint64_t dummy = 0;
            for (uint64_t j = 0; j < cycles; j++) dummy++;
            
            profile_fft_end(start, 1024, stage);
        }
    }
    
    // Check each stage converged to its optimal
    for (int stage = 0; stage < stages; stage++) {
        int final = get_tuned_distance(stage);
        bool near_optimal = abs(final - optimal[stage]) <= 4;
        
        char msg[64];
        snprintf(msg, sizeof(msg), "Stage %d converged (final=%d, opt=%d)", 
                 stage, final, optimal[stage]);
        ASSERT_TRUE(near_optimal, msg);
    }
    
    cleanup_adaptive_tuning();
}

void test_periodic_retuning(void) {
    printf("\n=== Test: Periodic Retuning ===\n");
    
    set_tuning_mode(TUNING_MODE_EWMA);
    configure_tuning(0.2, 5, 0.02, 20, 4);
    set_periodic_retune(true, 50); // Retune every 50 calls
    
    int stages = 1;
    int dist = 8;
    int ws = 64 * 1024;
    int radix = 4;
    
    init_adaptive_tuning(stages, &dist, &ws, &radix);
    
    // Run with optimal=12 for first 50 iterations
    for (int i = 0; i < 50; i++) {
        uint64_t start = profile_fft_start();
        int current = get_tuned_distance(0);
        uint64_t cycles = 10000 + abs(current - 12) * 500;
        
        volatile uint64_t dummy = 0;
        for (uint64_t j = 0; j < cycles; j++) dummy++;
        
        profile_fft_end(start, 1024, 0);
    }
    
    int dist_after_first = get_tuned_distance(0);
    
    // Now change optimal to 20 (workload shift!)
    for (int i = 0; i < 100; i++) {
        uint64_t start = profile_fft_start();
        int current = get_tuned_distance(0);
        uint64_t cycles = 10000 + abs(current - 20) * 500; // New optimal
        
        volatile uint64_t dummy = 0;
        for (uint64_t j = 0; j < cycles; j++) dummy++;
        
        profile_fft_end(start, 1024, 0);
    }
    
    int dist_after_shift = get_tuned_distance(0);
    
    // Should adapt to new workload
    ASSERT_TRUE(abs(dist_after_shift - 20) < abs(dist_after_first - 20),
                "Adapted to workload shift");
    
    printf("  After first: %d, After shift: %d (new optimal: 20)\n",
           dist_after_first, dist_after_shift);
    
    cleanup_adaptive_tuning();
}

//==============================================================================
// TEST SUITE 3: INTEGRATION TESTS
//==============================================================================

void test_throttle_tuning_integration(void) {
    printf("\n=== Test: Throttle + Tuning Integration ===\n");
    
    // Setup both systems
    set_throttle_mode(THROTTLE_MODE_TOKEN_BUCKET);
    configure_token_bucket(8, 12, 16, 1000, 4);
    set_throttle_statistics(true);
    init_throttling_enhanced(&test_cpu_profile);
    
    set_tuning_mode(TUNING_MODE_EWMA);
    configure_tuning(0.2, 5, 0.02, 20, 4);
    
    int stages = 1;
    int dist = 8;
    int ws = 64 * 1024;
    int radix = 4;
    init_adaptive_tuning(stages, &dist, &ws, &radix);
    
    // Run integrated workload
    for (int iter = 0; iter < 50; iter++) {
        uint64_t start = profile_fft_start();
        
        int distance = get_tuned_distance(0);
        
        // Simulate prefetch loop
        for (int i = 0; i < 100; i++) {
            prefetch_throttled_enhanced(
                (void*)(uintptr_t)(i + distance),
                _MM_HINT_T0,
                PREFETCH_PRIO_HIGH,
                do_prefetch
            );
        }
        
        // Simulate work
        uint64_t cycles = 10000 + abs(distance - 12) * 500;
        volatile uint64_t dummy = 0;
        for (uint64_t j = 0; j < cycles; j++) dummy++;
        
        profile_fft_end(start, 1024, 0);
    }
    
    // Check both systems functioned
    uint64_t req, iss, thr, crit;
    double rate;
    get_throttle_stats(&req, &iss, &thr, &crit, &rate);
    
    ASSERT_TRUE(req > 0, "Throttle system processed requests");
    ASSERT_TRUE(iss > 0, "Some prefetches issued");
    
    int final_dist = get_tuned_distance(0);
    ASSERT_TRUE(final_dist != dist, "Tuning adjusted distance");
    
    printf("  Throttle: %llu requests, %.1f%% throttled\n", req, rate * 100.0);
    printf("  Tuning: %d -> %d\n", dist, final_dist);
    
    cleanup_adaptive_tuning();
}

void test_autotune_token_bucket(void) {
    printf("\n=== Test: Auto-tune Token Bucket ===\n");
    
    set_throttle_mode(THROTTLE_MODE_TOKEN_BUCKET);
    configure_token_bucket(4, 6, 8, 1000, 2); // Start conservative
    set_throttle_statistics(true);
    init_throttling_enhanced(&test_cpu_profile);
    
    // Generate high throttle rate scenario
    for (int i = 0; i < 2000; i++) {
        prefetch_throttled_enhanced(
            (void*)(uintptr_t)i,
            _MM_HINT_T0,
            PREFETCH_PRIO_HIGH,
            do_prefetch
        );
    }
    
    uint64_t req1, iss1, thr1, crit1;
    double rate1;
    get_throttle_stats(&req1, &iss1, &thr1, &crit1, &rate1);
    
    // Auto-tune should increase capacity if rate > 30%
    autotune_token_bucket();
    
    // Reset stats and try again
    set_throttle_statistics(false);
    set_throttle_statistics(true);
    init_throttling_enhanced(&test_cpu_profile);
    
    for (int i = 0; i < 2000; i++) {
        prefetch_throttled_enhanced(
            (void*)(uintptr_t)i,
            _MM_HINT_T0,
            PREFETCH_PRIO_HIGH,
            do_prefetch
        );
    }
    
    uint64_t req2, iss2, thr2, crit2;
    double rate2;
    get_throttle_stats(&req2, &iss2, &thr2, &crit2, &rate2);
    
    // Throttle rate should improve
    ASSERT_TRUE(rate2 < rate1 || rate2 < 0.35, "Auto-tune improved or acceptable rate");
    
    printf("  Before: %.1f%% throttled, After: %.1f%%\n", rate1 * 100.0, rate2 * 100.0);
}

//==============================================================================
// TEST SUITE 4: EDGE CASES & ROBUSTNESS
//==============================================================================

void test_empty_tuning(void) {
    printf("\n=== Test: Empty Tuning (0 stages) ===\n");
    
    set_tuning_mode(TUNING_MODE_PER_STAGE);
    
    int stages = 0;
    init_adaptive_tuning(stages, NULL, NULL, NULL);
    
    // Should not crash
    uint64_t start = profile_fft_start();
    profile_fft_end(start, 1024, 0);
    
    int dist = get_tuned_distance(0);
    ASSERT_TRUE(dist < 0, "Returns invalid for out-of-bounds stage");
    
    cleanup_adaptive_tuning();
    printf("  Handled gracefully\n");
}

void test_negative_cycles(void) {
    printf("\n=== Test: Negative Cycles (time warp) ===\n");
    
    set_tuning_mode(TUNING_MODE_SIMPLE);
    configure_tuning(0.2, 5, 0.02, 20, 4);
    
    int stages = 1;
    int dist = 8;
    int ws = 64 * 1024;
    int radix = 4;
    init_adaptive_tuning(stages, &dist, &ws, &radix);
    
    // Simulate time going backwards (shouldn't crash)
    uint64_t start = 1000000;
    profile_fft_end(start, 1024, 0); // End before start
    
    printf("  Handled time warp gracefully\n");
    
    cleanup_adaptive_tuning();
}

void test_extreme_distances(void) {
    printf("\n=== Test: Extreme Distances ===\n");
    
    set_tuning_mode(TUNING_MODE_SIMPLE);
    configure_tuning(0.2, 5, 0.02, 20, 4);
    
    // Test with very small and very large distances
    int stages = 2;
    int dists[] = {1, 100};
    int ws[] = {1024, 1024*1024};
    int radixes[] = {2, 16};
    
    init_adaptive_tuning(stages, dists, ws, radixes);
    
    // Should clamp to valid range
    for (int i = 0; i < 20; i++) {
        uint64_t start = profile_fft_start();
        volatile uint64_t dummy = 0;
        for (int j = 0; j < 10000; j++) dummy++;
        profile_fft_end(start, 1024, 0);
    }
    
    int dist0 = get_tuned_distance(0);
    int dist1 = get_tuned_distance(1);
    
    ASSERT_TRUE(dist0 >= 2 && dist0 <= 64, "Distance 0 clamped to valid range");
    ASSERT_TRUE(dist1 >= 2 && dist1 <= 64, "Distance 1 clamped to valid range");
    
    printf("  Distances clamped: %d, %d\n", dist0, dist1);
    
    cleanup_adaptive_tuning();
}

void test_zero_elements(void) {
    printf("\n=== Test: Zero Elements ===\n");
    
    set_tuning_mode(TUNING_MODE_SIMPLE);
    configure_tuning(0.2, 5, 0.02, 20, 4);
    
    int stages = 1;
    int dist = 8;
    int ws = 64 * 1024;
    int radix = 4;
    init_adaptive_tuning(stages, &dist, &ws, &radix);
    
    // Profile with 0 elements (shouldn't divide by zero)
    uint64_t start = profile_fft_start();
    profile_fft_end(start, 0, 0);
    
    printf("  Handled zero elements gracefully\n");
    
    cleanup_adaptive_tuning();
}

void test_rapid_mode_switching(void) {
    printf("\n=== Test: Rapid Mode Switching ===\n");
    
    int stages = 1;
    int dist = 8;
    int ws = 64 * 1024;
    int radix = 4;
    
    // Switch modes rapidly
    for (int i = 0; i < 10; i++) {
        set_tuning_mode(TUNING_MODE_SIMPLE);
        init_adaptive_tuning(stages, &dist, &ws, &radix);
        cleanup_adaptive_tuning();
        
        set_tuning_mode(TUNING_MODE_EWMA);
        init_adaptive_tuning(stages, &dist, &ws, &radix);
        cleanup_adaptive_tuning();
        
        set_tuning_mode(TUNING_MODE_PER_STAGE);
        init_adaptive_tuning(stages, &dist, &ws, &radix);
        cleanup_adaptive_tuning();
    }
    
    printf("  Survived %d mode switches\n", 10 * 3);
}

//==============================================================================
// PERFORMANCE BENCHMARKS
//==============================================================================

void benchmark_throttle_overhead(void) {
    printf("\n=== Benchmark: Throttle Overhead ===\n");
    
    const int iterations = 1000000;
    uint64_t start, end;
    
    // Baseline: no throttling
    start = read_tsc();
    for (int i = 0; i < iterations; i++) {
        volatile int dummy = i;
    }
    end = read_tsc();
    uint64_t baseline = end - start;
    
    // Simple throttle
    set_throttle_mode(THROTTLE_MODE_SIMPLE);
    set_throttle_statistics(false);
    init_throttling_enhanced(&test_cpu_profile);
    
    start = read_tsc();
    for (int i = 0; i < iterations; i++) {
        prefetch_throttled_enhanced(
            (void*)(uintptr_t)i,
            _MM_HINT_T0,
            PREFETCH_PRIO_HIGH,
            do_prefetch
        );
    }
    end = read_tsc();
    uint64_t simple_cycles = end - start;
    
    // Token bucket
    set_throttle_mode(THROTTLE_MODE_TOKEN_BUCKET);
    init_throttling_enhanced(&test_cpu_profile);
    
    start = read_tsc();
    for (int i = 0; i < iterations; i++) {
        prefetch_throttled_enhanced(
            (void*)(uintptr_t)i,
            _MM_HINT_T0,
            PREFETCH_PRIO_HIGH,
            do_prefetch
        );
    }
    end = read_tsc();
    uint64_t token_cycles = end - start;
    
    double simple_overhead = (double)(simple_cycles - baseline) / iterations;
    double token_overhead = (double)(token_cycles - baseline) / iterations;
    
    printf("  Baseline: %llu cycles\n", baseline);
    printf("  Simple throttle: %llu cycles (%.2f per call)\n", 
           simple_cycles, simple_overhead);
    printf("  Token bucket: %llu cycles (%.2f per call)\n",
           token_cycles, token_overhead);
    printf("  Overhead: Simple=%.1f%%, Token=%.1f%%\n",
           100.0 * simple_overhead / (baseline / (double)iterations),
           100.0 * token_overhead / (baseline / (double)iterations));
}

void benchmark_tuning_overhead(void) {
    printf("\n=== Benchmark: Tuning Overhead ===\n");
    
    const int iterations = 10000;
    uint64_t start, end;
    
    int stages = 1;
    int dist = 8;
    int ws = 64 * 1024;
    int radix = 4;
    
    // No tuning
    set_tuning_mode(TUNING_MODE_DISABLED);
    init_adaptive_tuning(stages, &dist, &ws, &radix);
    
    start = read_tsc();
    for (int i = 0; i < iterations; i++) {
        uint64_t s = profile_fft_start();
        volatile int dummy = 0;
        for (int j = 0; j < 1000; j++) dummy++;
        profile_fft_end(s, 1024, 0);
    }
    end = read_tsc();
    uint64_t disabled_cycles = end - start;
    
    cleanup_adaptive_tuning();
    
    // Simple tuning
    set_tuning_mode(TUNING_MODE_SIMPLE);
    init_adaptive_tuning(stages, &dist, &ws, &radix);
    
    start = read_tsc();
    for (int i = 0; i < iterations; i++) {
        uint64_t s = profile_fft_start();
        volatile int dummy = 0;
        for (int j = 0; j < 1000; j++) dummy++;
        profile_fft_end(s, 1024, 0);
    }
    end = read_tsc();
    uint64_t simple_cycles = end - start;
    
    cleanup_adaptive_tuning();
    
    // Per-stage tuning
    set_tuning_mode(TUNING_MODE_PER_STAGE);
    init_adaptive_tuning(stages, &dist, &ws, &radix);
    
    start = read_tsc();
    for (int i = 0; i < iterations; i++) {
        uint64_t s = profile_fft_start();
        volatile int dummy = 0;
        for (int j = 0; j < 1000; j++) dummy++;
        profile_fft_end(s, 1024, 0);
    }
    end = read_tsc();
    uint64_t per_stage_cycles = end - start;
    
    cleanup_adaptive_tuning();
    
    double simple_overhead = (double)(simple_cycles - disabled_cycles) / iterations;
    double per_stage_overhead = (double)(per_stage_cycles - disabled_cycles) / iterations;
    
    printf("  Disabled: %llu cycles\n", disabled_cycles);
    printf("  Simple: %llu cycles (+%.2f per call)\n", 
           simple_cycles, simple_overhead);
    printf("  Per-stage: %llu cycles (+%.2f per call)\n",
           per_stage_cycles, per_stage_overhead);
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

void run_all_tests(void) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     PREFETCH SYSTEM TEST SUITE                             ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    
    // Throttling tests
    printf("\n--- THROTTLING TESTS ---\n");
    test_simple_throttle_basic();
    test_critical_priority_bypass();
    test_token_bucket_refill();
    test_per_level_token_buckets();
    
    // Tuning tests
    printf("\n--- ADAPTIVE TUNING TESTS ---\n");
    test_ewma_filter();
    test_simple_tuning_convergence();
    test_per_stage_independence();
    test_periodic_retuning();
    
    // Integration tests
    printf("\n--- INTEGRATION TESTS ---\n");
    test_throttle_tuning_integration();
    test_autotune_token_bucket();
    
    // Edge cases
    printf("\n--- EDGE CASE TESTS ---\n");
    test_empty_tuning();
    test_negative_cycles();
    test_extreme_distances();
    test_zero_elements();
    test_rapid_mode_switching();
    
    // Performance benchmarks
    printf("\n--- PERFORMANCE BENCHMARKS ---\n");
    benchmark_throttle_overhead();
    benchmark_tuning_overhead();
    
    // Summary
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║     TEST SUMMARY                                           ║\n");
    printf("╠════════════════════════════════════════════════════════════╣\n");
    printf("║  Total tests: %-4d                                        ║\n", g_tests_run);
    printf("║  Passed:      %-4d                                        ║\n", g_tests_passed);
    printf("║  Failed:      %-4d                                        ║\n", g_tests_run - g_tests_passed);
    printf("║  Success rate: %3.0f%%                                      ║\n", 
           100.0 * g_tests_passed / g_tests_run);
    printf("╚════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

int main(int argc, char **argv) {
    // Run tests
    run_all_tests();
    
    // Return failure if any tests failed
    return (g_tests_run == g_tests_passed) ? 0 : 1;
}
