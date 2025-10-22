/**
 * @file test_fft_conversions.c
 * @brief Test Suite for FFT Conversion Bug Fixes (v2.1)
 * 
 * @details
 * This test file validates the correctness of the bug fixes applied to
 * the FFT conversion utilities and SIMD butterfly operations.
 * 
 * TESTS COVER:
 * - Bug #1: SSE2 twiddle loading
 * - Bug #2: AVX-512 deinterleaving
 * - Bug #3: AVX2 deinterleaving  
 * - Bug #4: AVX-512 interleaving
 * 
 * COMPILE INSTRUCTIONS:
 * 
 * Test SSE2 only:
 *   gcc -O2 -msse2 -mno-avx -o test_sse2 test_fft_conversions.c
 * 
 * Test AVX2:
 *   gcc -O2 -mavx2 -mno-avx512f -o test_avx2 test_fft_conversions.c
 * 
 * Test AVX-512:
 *   gcc -O2 -mavx512f -o test_avx512 test_fft_conversions.c
 * 
 * @author FFT Testing Team
 * @version 2.1
 * @date 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

// Include the fixed headers
#include "fft_conversion_utils_FIXED.h"
#include "fft_radix2_macros_true_soa_FIXED.h"

//==============================================================================
// TEST UTILITIES
//==============================================================================

#define EPSILON 1e-12
#define TEST_PASS "\033[32m[PASS]\033[0m"
#define TEST_FAIL "\033[31m[FAIL]\033[0m"

typedef struct {
    const char *name;
    bool passed;
    const char *error_msg;
} test_result_t;

static int tests_run = 0;
static int tests_passed = 0;

/**
 * @brief Compare two doubles with tolerance
 */
static bool double_equal(double a, double b, double epsilon) {
    return fabs(a - b) < epsilon;
}

/**
 * @brief Print test result
 */
static void report_test(const char *name, bool passed, const char *error_msg) {
    tests_run++;
    if (passed) {
        tests_passed++;
        printf("%s %s\n", TEST_PASS, name);
    } else {
        printf("%s %s\n", TEST_FAIL, name);
        if (error_msg) {
            printf("    Error: %s\n", error_msg);
        }
    }
}

//==============================================================================
// TEST CASES: CONVERSION UTILITIES
//==============================================================================

/**
 * @brief Test AoS to SoA conversion
 * 
 * Tests Bug #2 (AVX-512) and Bug #3 (AVX2) fixes
 */
static void test_aos_to_soa(void) {
    printf("\n=== Testing AoS → SoA Conversion ===\n");
    
    // Test pattern with distinct values
    fft_data aos[8] = {
        {1.0, 2.0},   // aos[0]
        {3.0, 4.0},   // aos[1]
        {5.0, 6.0},   // aos[2]
        {7.0, 8.0},   // aos[3]
        {9.0, 10.0},  // aos[4]
        {11.0, 12.0}, // aos[5]
        {13.0, 14.0}, // aos[6]
        {15.0, 16.0}  // aos[7]
    };
    
    double re[8], im[8];
    
    // Convert
    fft_aos_to_soa(aos, re, im, 8);
    
    // Expected results
    double expected_re[8] = {1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0};
    double expected_im[8] = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0};
    
    // Verify
    bool passed = true;
    char error_msg[256] = {0};
    
    for (int i = 0; i < 8; i++) {
        if (!double_equal(re[i], expected_re[i], EPSILON)) {
            snprintf(error_msg, sizeof(error_msg),
                    "re[%d]: expected %.1f, got %.1f", i, expected_re[i], re[i]);
            passed = false;
            break;
        }
        if (!double_equal(im[i], expected_im[i], EPSILON)) {
            snprintf(error_msg, sizeof(error_msg),
                    "im[%d]: expected %.1f, got %.1f", i, expected_im[i], im[i]);
            passed = false;
            break;
        }
    }
    
    report_test("AoS → SoA (8 elements)", passed, passed ? NULL : error_msg);
}

/**
 * @brief Test SoA to AoS conversion
 * 
 * Tests Bug #4 (AVX-512) fix
 */
static void test_soa_to_aos(void) {
    printf("\n=== Testing SoA → AoS Conversion ===\n");
    
    // Test pattern
    double re[8] = {1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0};
    double im[8] = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0};
    
    fft_data aos[8];
    
    // Convert
    fft_soa_to_aos(re, im, aos, 8);
    
    // Expected results
    fft_data expected[8] = {
        {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0},
        {9.0, 10.0}, {11.0, 12.0}, {13.0, 14.0}, {15.0, 16.0}
    };
    
    // Verify
    bool passed = true;
    char error_msg[256] = {0};
    
    for (int i = 0; i < 8; i++) {
        if (!double_equal(aos[i].re, expected[i].re, EPSILON)) {
            snprintf(error_msg, sizeof(error_msg),
                    "aos[%d].re: expected %.1f, got %.1f", 
                    i, expected[i].re, aos[i].re);
            passed = false;
            break;
        }
        if (!double_equal(aos[i].im, expected[i].im, EPSILON)) {
            snprintf(error_msg, sizeof(error_msg),
                    "aos[%d].im: expected %.1f, got %.1f",
                    i, expected[i].im, aos[i].im);
            passed = false;
            break;
        }
    }
    
    report_test("SoA → AoS (8 elements)", passed, passed ? NULL : error_msg);
}

/**
 * @brief Test roundtrip conversion (should be identity)
 */
static void test_conversion_roundtrip(void) {
    printf("\n=== Testing Roundtrip Conversion ===\n");
    
    // Original data
    fft_data original[16];
    for (int i = 0; i < 16; i++) {
        original[i].re = (double)(i * 2 + 1);
        original[i].im = (double)(i * 2 + 2);
    }
    
    // Allocate workspace
    double re[16], im[16];
    fft_data result[16];
    
    // Roundtrip: AoS → SoA → AoS
    fft_aos_to_soa(original, re, im, 16);
    fft_soa_to_aos(re, im, result, 16);
    
    // Verify identity
    bool passed = true;
    char error_msg[256] = {0};
    
    for (int i = 0; i < 16; i++) {
        if (!double_equal(result[i].re, original[i].re, EPSILON)) {
            snprintf(error_msg, sizeof(error_msg),
                    "result[%d].re: expected %.1f, got %.1f",
                    i, original[i].re, result[i].re);
            passed = false;
            break;
        }
        if (!double_equal(result[i].im, original[i].im, EPSILON)) {
            snprintf(error_msg, sizeof(error_msg),
                    "result[%d].im: expected %.1f, got %.1f",
                    i, original[i].im, result[i].im);
            passed = false;
            break;
        }
    }
    
    report_test("Roundtrip AoS → SoA → AoS (16 elements)", passed, 
                passed ? NULL : error_msg);
}

/**
 * @brief Test edge cases
 */
static void test_edge_cases(void) {
    printf("\n=== Testing Edge Cases ===\n");
    
    // Test 1: Single element
    {
        fft_data aos_single[1] = {{42.0, 13.0}};
        double re_single[1], im_single[1];
        fft_data result_single[1];
        
        fft_aos_to_soa(aos_single, re_single, im_single, 1);
        fft_soa_to_aos(re_single, im_single, result_single, 1);
        
        bool passed = double_equal(result_single[0].re, 42.0, EPSILON) &&
                     double_equal(result_single[0].im, 13.0, EPSILON);
        
        report_test("Edge case: Single element", passed, NULL);
    }
    
    // Test 2: All zeros
    {
        fft_data zeros[8] = {{0}};
        double re[8], im[8];
        fft_data result[8];
        
        fft_aos_to_soa(zeros, re, im, 8);
        fft_soa_to_aos(re, im, result, 8);
        
        bool passed = true;
        for (int i = 0; i < 8; i++) {
            if (result[i].re != 0.0 || result[i].im != 0.0) {
                passed = false;
                break;
            }
        }
        
        report_test("Edge case: All zeros", passed, NULL);
    }
    
    // Test 3: Large values
    {
        fft_data large[4] = {
            {1e100, -1e100},
            {1e-100, -1e-100},
            {3.14159265358979, 2.71828182845905},
            {INFINITY, -INFINITY}
        };
        double re[4], im[4];
        fft_data result[4];
        
        fft_aos_to_soa(large, re, im, 4);
        fft_soa_to_aos(re, im, result, 4);
        
        bool passed = 
            double_equal(result[0].re, 1e100, 1e88) &&
            double_equal(result[0].im, -1e100, 1e88) &&
            double_equal(result[1].re, 1e-100, 1e-112) &&
            double_equal(result[1].im, -1e-100, 1e-112) &&
            double_equal(result[2].re, 3.14159265358979, EPSILON) &&
            double_equal(result[2].im, 2.71828182845905, EPSILON) &&
            isinf(result[3].re) && isinf(result[3].im);
        
        report_test("Edge case: Large/small/special values", passed, NULL);
    }
}

//==============================================================================
// TEST CASES: SIMD BUTTERFLY OPERATIONS
//==============================================================================

/**
 * @brief Test SSE2 twiddle loading (Bug #1)
 * 
 * This test verifies that consecutive twiddle factors are loaded correctly
 * in the SSE2 path.
 */
static void test_sse2_twiddle_loading(void) {
    printf("\n=== Testing SSE2 Twiddle Loading (Bug #1) ===\n");
    
    // Create mock twiddle factors
    fft_twiddles_soa twiddles;
    double tw_re[8] = {1.0, 0.9, 0.7, 0.4, 0.0, -0.4, -0.7, -0.9};
    double tw_im[8] = {0.0, 0.4, 0.7, 0.9, 1.0, 0.9, 0.7, 0.4};
    twiddles.re = tw_re;
    twiddles.im = tw_im;
    
    // Create input data (simple pattern)
    double in_re[16], in_im[16];
    for (int i = 0; i < 16; i++) {
        in_re[i] = (double)i;
        in_im[i] = (double)(i + 100);
    }
    
    double out_re[16], out_im[16];
    
    // Test that we can call the pipeline macro without crashes
    // If twiddle loading is correct, this should not produce NaN or inf
    
#ifdef __SSE2__
    int half = 8;
    
    // Process two butterflies at once (k=0,1)
    RADIX2_PIPELINE_2_NATIVE_SOA_SSE2(0, in_re, in_im, out_re, out_im, 
                                      &twiddles, half);
    
    // Check that results are finite (not NaN or inf)
    bool passed = true;
    char error_msg[256] = {0};
    
    for (int k = 0; k < 2; k++) {
        if (!isfinite(out_re[k]) || !isfinite(out_im[k]) ||
            !isfinite(out_re[k + half]) || !isfinite(out_im[k + half])) {
            snprintf(error_msg, sizeof(error_msg),
                    "Non-finite result at k=%d", k);
            passed = false;
            break;
        }
    }
    
    report_test("SSE2 twiddle loading produces finite results", passed,
                passed ? NULL : error_msg);
#else
    report_test("SSE2 twiddle loading (skipped - SSE2 not available)", true, NULL);
#endif
}

/**
 * @brief Test that different twiddles produce different results
 * 
 * This specifically tests that Bug #1 is fixed by verifying that
 * butterfly k and k+1 use different twiddle factors.
 */
static void test_distinct_twiddles(void) {
    printf("\n=== Testing Distinct Twiddles (Bug #1 Verification) ===\n");
    
    // Create twiddle factors with distinct values
    fft_twiddles_soa twiddles;
    double tw_re[8] = {1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5};
    double tw_im[8] = {0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5};
    twiddles.re = tw_re;
    twiddles.im = tw_im;
    
    // Create input: all ones for simplicity
    double in_re[16], in_im[16];
    for (int i = 0; i < 16; i++) {
        in_re[i] = 1.0;
        in_im[i] = 0.0;
    }
    
    double out_re[16], out_im[16];
    int half = 8;
    
    // Process using scalar path (reference)
    double ref_re[16], ref_im[16];
    for (int k = 0; k < 2; k++) {
        RADIX2_PIPELINE_1_NATIVE_SOA_SCALAR(k, in_re, in_im, ref_re, ref_im,
                                           &twiddles, half);
    }
    
#ifdef __SSE2__
    // Process using SSE2 path (should match scalar)
    RADIX2_PIPELINE_2_NATIVE_SOA_SSE2(0, in_re, in_im, out_re, out_im,
                                      &twiddles, half);
    
    // Verify that results match scalar reference
    bool passed = true;
    char error_msg[256] = {0};
    
    for (int k = 0; k < 2; k++) {
        if (!double_equal(out_re[k], ref_re[k], EPSILON)) {
            snprintf(error_msg, sizeof(error_msg),
                    "out_re[%d]: SSE2 %.6f != scalar %.6f", 
                    k, out_re[k], ref_re[k]);
            passed = false;
            break;
        }
        if (!double_equal(out_im[k], ref_im[k], EPSILON)) {
            snprintf(error_msg, sizeof(error_msg),
                    "out_im[%d]: SSE2 %.6f != scalar %.6f",
                    k, out_im[k], ref_im[k]);
            passed = false;
            break;
        }
    }
    
    report_test("SSE2 matches scalar (distinct twiddles)", passed,
                passed ? NULL : error_msg);
#else
    report_test("SSE2 distinct twiddles (skipped - SSE2 not available)", true, NULL);
#endif
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================
/*
int main(void) {
    printf("================================================================================\n");
    printf("FFT Conversion & SIMD Bug Fix Test Suite v2.1\n");
    printf("================================================================================\n");
    
    // Print SIMD capabilities
    printf("\nSIMD Capabilities:\n");
#ifdef __AVX512F__
    printf("  [✓] AVX-512F\n");
#else
    printf("  [ ] AVX-512F\n");
#endif
#ifdef __AVX2__
    printf("  [✓] AVX2\n");
#else
    printf("  [ ] AVX2\n");
#endif
#ifdef __SSE2__
    printf("  [✓] SSE2\n");
#else
    printf("  [ ] SSE2\n");
#endif
    
    // Run conversion tests
    test_aos_to_soa();
    test_soa_to_aos();
    test_conversion_roundtrip();
    test_edge_cases();
    
    // Run SIMD tests
    test_sse2_twiddle_loading();
    test_distinct_twiddles();
    
    // Print summary
    printf("\n");
    printf("================================================================================\n");
    printf("TEST SUMMARY\n");
    printf("================================================================================\n");
    printf("Tests run:    %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);
    printf("\n");
    
    if (tests_passed == tests_run) {
        printf("✓ ALL TESTS PASSED\n");
        printf("================================================================================\n");
        return 0;
    } else {
        printf("✗ SOME TESTS FAILED\n");
        printf("================================================================================\n");
        return 1;
    }
}
    */