#include "highspeedFFT.h"
#include "real.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <cpuid.h>
#include <time.h>

// For _mm_malloc/_mm_free
#if defined(_MSC_VER)
    #include <malloc.h>
#elif defined(__GNUC__) || defined(__clang__)
    #include <mm_malloc.h>
#endif

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

//==============================================================================
// UTILITY FUNCTIONS
//==============================================================================

void generate_signal(fft_data *signal, int length, double freq, double amplitude)
{
    for (int i = 0; i < length; i++)
    {
        signal[i].re = amplitude * sin(2.0 * M_PI * freq * i / length);
        signal[i].im = 0.0;
    }
}

double compute_mse(fft_data *original, fft_data *reconstructed, int length)
{
    double mse = 0.0;
    for (int i = 0; i < length; i++)
    {
        double diff_re = original[i].re - reconstructed[i].re;
        double diff_im = original[i].im - reconstructed[i].im;
        mse += diff_re * diff_re + diff_im * diff_im;
    }
    return mse / length;
}

void print_complex(fft_data *data, int length, const char *label)
{
    printf("%s:\n", label);
    for (int i = 0; i < length && i < 10; i++)
    {
        printf("  [%d] %.6f + %.6fi\n", i, data[i].re, data[i].im);
    }
    if (length > 10) printf("  ... (%d more)\n", length - 10);
    printf("\n");
}

// Benchmark helper
double benchmark_fft(fft_object fft, fft_data *input, fft_data *output, int N, int iterations)
{
    clock_t start = clock();
    for (int i = 0; i < iterations; i++)
    {
        fft_exec(fft, input, output);
    }
    clock_t end = clock();
    
    return (double)(end - start) / CLOCKS_PER_SEC / iterations;
}

//==============================================================================
// CPU FEATURE DETECTION
//==============================================================================

void detect_cpu_features(void)
{
    unsigned int eax, ebx, ecx, edx;
    
    printf("===========================================\n");
    printf("CPU SIMD Feature Detection\n");
    printf("===========================================\n");
    
    // Check CPUID leaf 1 (Processor Info and Feature Bits)
    __cpuid(1, eax, ebx, ecx, edx);
    
    printf("SSE:      %s\n", (edx & (1 << 25)) ? "YES ✓" : "NO ✗");
    printf("SSE2:     %s\n", (edx & (1 << 26)) ? "YES ✓" : "NO ✗");
    printf("SSE3:     %s\n", (ecx & (1 << 0))  ? "YES ✓" : "NO ✗");
    printf("SSSE3:    %s\n", (ecx & (1 << 9))  ? "YES ✓" : "NO ✗");
    printf("SSE4.1:   %s\n", (ecx & (1 << 19)) ? "YES ✓" : "NO ✗");
    printf("SSE4.2:   %s\n", (ecx & (1 << 20)) ? "YES ✓" : "NO ✗");
    printf("AVX:      %s\n", (ecx & (1 << 28)) ? "YES ✓" : "NO ✗");
    printf("FMA3:     %s\n", (ecx & (1 << 12)) ? "YES ✓" : "NO ✗");
    
    // Check CPUID leaf 7 (Extended Features)
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    
    printf("AVX2:     %s\n", (ebx & (1 << 5))  ? "YES ✓" : "NO ✗");
    printf("AVX-512F: %s\n", (ebx & (1 << 16)) ? "YES ✓" : "NO ✗");
    
    printf("\n");
    
    // Detect compile-time flags
    printf("Compiled with:\n");
#ifdef __AVX2__
    printf("  __AVX2__ ✓\n");
#else
    printf("  __AVX2__ ✗\n");
#endif
#ifdef __FMA__
    printf("  __FMA__ ✓\n");
#else
    printf("  __FMA__ ✗\n");
#endif
#ifdef HAS_AVX512
    printf("  HAS_AVX512 ✓ (but not supported by your CPU)\n");
#endif
    
    printf("===========================================\n\n");
}

//==============================================================================
// MAIN TEST PROGRAM
//==============================================================================

int main(void)
{
    detect_cpu_features();
    
    printf("===========================================\n");
    printf("FFT Library Test Suite - AVX2 Optimized\n");
    printf("Intel Core i9-14900KF Performance Tests\n");
    printf("===========================================\n\n");

    const double mse_tolerance = 1e-10;

    //==========================================================================
    // POWER-OF-2 TESTS (Radix-2/4/8/16/32 with AVX2)
    //==========================================================================
    printf("=== Power-of-2 FFT Tests (AVX2 Optimized) ===\n\n");
    
    int power2_lengths[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int num_power2_tests = sizeof(power2_lengths) / sizeof(power2_lengths[0]);

    int power2_passed = 0;
    int power2_failed = 0;

    for (int test = 0; test < num_power2_tests; test++)
    {
        int N = power2_lengths[test];
        printf("Testing Power-of-2 FFT with N = %d\n", N);

        // Allocate aligned memory for better AVX2 performance
        fft_data *input = (fft_data *)_mm_malloc(N * sizeof(fft_data), 32);
        fft_data *output = (fft_data *)_mm_malloc(N * sizeof(fft_data), 32);
        fft_data *inverse = (fft_data *)_mm_malloc(N * sizeof(fft_data), 32);
        
        if (!input || !output || !inverse)
        {
            fprintf(stderr, "  ERROR: Memory allocation failed for N = %d\n", N);
            if (input) _mm_free(input);
            if (output) _mm_free(output);
            if (inverse) _mm_free(inverse);
            power2_failed++;
            continue;
        }

        // Generate test signal (multi-tone)
        for (int i = 0; i < N; i++)
        {
            input[i].re = sin(2.0 * M_PI * 5.0 * i / N) + 
                          0.5 * cos(2.0 * M_PI * 13.0 * i / N) +
                          0.25 * sin(2.0 * M_PI * 23.0 * i / N);
            input[i].im = 0.0;
        }

        // Initialize FFT objects
        fft_object fft = fft_init(N, 1);
        if (!fft)
        {
            fprintf(stderr, "  ERROR: FFT initialization failed for N = %d\n", N);
            _mm_free(input);
            _mm_free(output);
            _mm_free(inverse);
            power2_failed++;
            continue;
        }
        
        fft_object ifft = fft_init(N, -1);
        if (!ifft)
        {
            fprintf(stderr, "  ERROR: Inverse FFT initialization failed for N = %d\n", N);
            free_fft(fft);
            _mm_free(input);
            _mm_free(output);
            _mm_free(inverse);
            power2_failed++;
            continue;
        }

        printf("  Algorithm: %s\n", fft->lt == 0 ? "Mixed-Radix" : "Bluestein");
        
        // Benchmark
        int iterations = (N <= 1024) ? 1000 : 100;
        double avg_time = benchmark_fft(fft, input, output, N, iterations);
        printf("  Avg time: %.6f ms (%.2f MFFT/s)\n", 
               avg_time * 1000.0, 
               1.0 / (avg_time * 1000.0));
        
        // Forward FFT
        fft_exec(fft, input, output);

        // Inverse FFT
        fft_exec(ifft, output, inverse);

        // Scale inverse
        for (int i = 0; i < N; i++)
        {
            inverse[i].re /= N;
            inverse[i].im /= N;
        }

        // Compute MSE
        double mse = compute_mse(input, inverse, N);
        printf("  MSE: %.6e ", mse);
        
        if (mse < mse_tolerance)
        {
            printf("✓ PASSED\n");
            power2_passed++;
        }
        else
        {
            printf("✗ FAILED (MSE exceeds tolerance)\n");
            power2_failed++;
            
            // Debug output
            printf("  First 4 values:\n");
            for (int i = 0; i < 4 && i < N; i++)
            {
                printf("    Input[%d]:  %.6f + %.6fi\n", i, input[i].re, input[i].im);
                printf("    Output[%d]: %.6f + %.6fi\n", i, inverse[i].re, inverse[i].im);
                printf("    Error[%d]:  %.6e\n", i, 
                       sqrt((input[i].re - inverse[i].re) * (input[i].re - inverse[i].re) +
                            (input[i].im - inverse[i].im) * (input[i].im - inverse[i].im)));
            }
        }

        // Cleanup
        free_fft(fft);
        free_fft(ifft);
        _mm_free(input);
        _mm_free(output);
        _mm_free(inverse);
        
        printf("\n");
    }
    
    printf("===========================================\n");
    printf("Power-of-2 Test Summary: %d passed, %d failed\n", power2_passed, power2_failed);
    printf("===========================================\n\n");

    //==========================================================================
    // MIXED-RADIX TESTS (Radix-3/5/7/11/13 with AVX2)
    //==========================================================================
    printf("=== Mixed-Radix FFT Tests (AVX2 Optimized) ===\n\n");
    
    int mixed_radix_lengths[] = {12, 15, 20, 28, 35, 60, 63, 77, 80, 120};
    int num_mixed_tests = sizeof(mixed_radix_lengths) / sizeof(mixed_radix_lengths[0]);
    
    int mixed_passed = 0;
    int mixed_failed = 0;

    for (int test = 0; test < num_mixed_tests; test++)
    {
        int N = mixed_radix_lengths[test];
        printf("Testing Mixed-Radix FFT with N = %d\n", N);

        // Allocate aligned memory
        fft_data *input = (fft_data *)_mm_malloc(N * sizeof(fft_data), 32);
        fft_data *output = (fft_data *)_mm_malloc(N * sizeof(fft_data), 32);
        fft_data *inverse = (fft_data *)_mm_malloc(N * sizeof(fft_data), 32);
        
        if (!input || !output || !inverse)
        {
            fprintf(stderr, "  ERROR: Memory allocation failed\n");
            if (input) _mm_free(input);
            if (output) _mm_free(output);
            if (inverse) _mm_free(inverse);
            mixed_failed++;
            continue;
        }

        // Generate test signal
        generate_signal(input, N, 2.0, 1.0);

        // Initialize FFT
        fft_object fft = fft_init(N, 1);
        fft_object ifft = fft_init(N, -1);
        
        if (!fft || !ifft)
        {
            fprintf(stderr, "  ERROR: FFT initialization failed\n");
            if (fft) free_fft(fft);
            if (ifft) free_fft(ifft);
            _mm_free(input);
            _mm_free(output);
            _mm_free(inverse);
            mixed_failed++;
            continue;
        }

        // Execute FFT
        fft_exec(fft, input, output);
        fft_exec(ifft, output, inverse);

        // Scale
        for (int i = 0; i < N; i++)
        {
            inverse[i].re /= N;
            inverse[i].im /= N;
        }

        // Check error
        double mse = compute_mse(input, inverse, N);
        printf("  MSE: %.6e ", mse);
        
        if (mse < mse_tolerance)
        {
            printf("✓ PASSED\n");
            mixed_passed++;
        }
        else
        {
            printf("✗ FAILED\n");
            mixed_failed++;
        }

        // Cleanup
        free_fft(fft);
        free_fft(ifft);
        _mm_free(input);
        _mm_free(output);
        _mm_free(inverse);
    }
    
    printf("\n===========================================\n");
    printf("Mixed-Radix Test Summary: %d passed, %d failed\n", mixed_passed, mixed_failed);
    printf("===========================================\n\n");

    //==========================================================================
    // BLUESTEIN TESTS (Prime sizes)
    //==========================================================================
    printf("=== Bluestein FFT Tests (Prime Sizes) ===\n\n");
    
    int prime_lengths[] = {7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
    int num_prime_tests = sizeof(prime_lengths) / sizeof(prime_lengths[0]);
    
    int prime_passed = 0;
    int prime_failed = 0;

    for (int test = 0; test < num_prime_tests; test++)
    {
        int N = prime_lengths[test];
        printf("Testing Bluestein FFT with N = %d\n", N);

        fft_data *input = (fft_data *)_mm_malloc(N * sizeof(fft_data), 32);
        fft_data *output = (fft_data *)_mm_malloc(N * sizeof(fft_data), 32);
        fft_data *inverse = (fft_data *)_mm_malloc(N * sizeof(fft_data), 32);
        
        if (!input || !output || !inverse)
        {
            fprintf(stderr, "  ERROR: Memory allocation failed\n");
            if (input) _mm_free(input);
            if (output) _mm_free(output);
            if (inverse) _mm_free(inverse);
            prime_failed++;
            continue;
        }

        generate_signal(input, N, 1.0, 1.0);

        fft_object fft = fft_init(N, 1);
        fft_object ifft = fft_init(N, -1);
        
        if (!fft || !ifft)
        {
            fprintf(stderr, "  ERROR: FFT initialization failed\n");
            if (fft) free_fft(fft);
            if (ifft) free_fft(ifft);
            _mm_free(input);
            _mm_free(output);
            _mm_free(inverse);
            prime_failed++;
            continue;
        }

        fft_exec(fft, input, output);
        fft_exec(ifft, output, inverse);

        for (int i = 0; i < N; i++)
        {
            inverse[i].re /= N;
            inverse[i].im /= N;
        }

        double mse = compute_mse(input, inverse, N);
        printf("  MSE: %.6e ", mse);
        
        if (mse < mse_tolerance)
        {
            printf("✓ PASSED\n");
            prime_passed++;
        }
        else
        {
            printf("✗ FAILED\n");
            prime_failed++;
        }

        free_fft(fft);
        free_fft(ifft);
        _mm_free(input);
        _mm_free(output);
        _mm_free(inverse);
    }
    
    printf("\n===========================================\n");
    printf("Bluestein Test Summary: %d passed, %d failed\n", prime_passed, prime_failed);
    printf("===========================================\n\n");

    //==========================================================================
    // FINAL SUMMARY
    //==========================================================================
    int total_passed = power2_passed + mixed_passed + prime_passed;
    int total_failed = power2_failed + mixed_failed + prime_failed;
    
    printf("===========================================\n");
    printf("OVERALL TEST SUMMARY\n");
    printf("===========================================\n");
    printf("Total Passed: %d\n", total_passed);
    printf("Total Failed: %d\n", total_failed);
    printf("Success Rate: %.1f%%\n", 
           total_passed * 100.0 / (total_passed + total_failed));
    printf("===========================================\n");

    return (total_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}