#include "highspeedFFT.h"
#include "real.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <cpuid.h>

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
    for (int i = 0; i < length && i < 10; i++)  // Limit to 10 for brevity
    {
        printf("  [%d] %.6f + %.6fi\n", i, data[i].re, data[i].im);
    }
    if (length > 10) printf("  ... (%d more)\n", length - 10);
    printf("\n");
}

//==============================================================================
// MAIN TEST PROGRAM
//==============================================================================

int main(void)
{

    unsigned int eax, ebx, ecx, edx;
    
    // Check CPUID leaf 7, subleaf 0 (Extended Features)
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    
    printf("=== AVX-512 Feature Detection ===\n");
    printf("AVX-512F (Foundation):      %s\n", (ebx & (1 << 16)) ? "YES ✓" : "NO ✗");
    printf("AVX-512DQ (Doubleword/Quad): %s\n", (ebx & (1 << 17)) ? "YES ✓" : "NO ✗");
    printf("AVX-512BW (Byte/Word):      %s\n", (ebx & (1 << 30)) ? "YES ✓" : "NO ✗");
    printf("AVX-512VL (Vector Length):  %s\n", (ebx & (1 << 31)) ? "YES ✓" : "NO ✗");
    
    if (ebx & (1 << 16)) {
        printf("\n✓ AVX-512 is ENABLED and ready to use!\n");
    } else {
        printf("\n✗ AVX-512 is DISABLED. Check BIOS settings.\n");
    }

    printf("===========================================\n");
    printf("FFT Library Test Suite\n");
    printf("===========================================\n\n");

    const double mse_tolerance = 1e-10;

#ifdef HAS_AVX512
    printf("=== AVX-512 Radix-2 Tests ===\n");
    printf("AVX-512 support detected!\n\n");
    
    int radix2_lengths[] = {64, 128, 256, 512, 1024, 2048};
    int num_radix2_tests = sizeof(radix2_lengths) / sizeof(radix2_lengths[0]);

    int passed = 0;
    int failed = 0;

    for (int test = 0; test < num_radix2_tests; test++)
    {
        int N = radix2_lengths[test];
        printf("Testing AVX-512 Radix-2 FFT with N = %d\n", N);

        // Allocate aligned memory
        fft_data *input = (fft_data *)_mm_malloc(N * sizeof(fft_data), 64);
        fft_data *output = (fft_data *)_mm_malloc(N * sizeof(fft_data), 64);
        fft_data *inverse = (fft_data *)_mm_malloc(N * sizeof(fft_data), 64);
        
        if (!input || !output || !inverse)
        {
          //  fprintf(stderr, "  ERROR: Memory allocation failed for N = %d\n", N);
            printf("boom");
            if (input) _mm_free(input);
            if (output) _mm_free(output);
            if (inverse) _mm_free(inverse);
            failed++;
            continue;
        }
         printf("boom2");
        // Generate test signal
        for (int i = 0; i < N; i++)
        {
            input[i].re = sin(2.0 * M_PI * 5.0 * i / N) + 
                          0.5 * cos(2.0 * M_PI * 13.0 * i / N);
            input[i].im = 0.0;
        }

        // Initialize FFT objects
        fft_object fft = fft_init(N, 1);
          printf("boom3");
        if (!fft)
        {
         //   fprintf(stderr, "  ERROR: FFT initialization failed for N = %d\n", N);
            _mm_free(input);
            _mm_free(output);
            _mm_free(inverse);
            failed++;
            continue;
        }
        
        fft_object ifft = fft_init(N, -1);
         printf("boom4");
        if (!ifft)
        {
         //   fprintf(stderr, "  ERROR: Inverse FFT initialization failed for N = %d\n", N);
            free_fft(fft);
            _mm_free(input);
            _mm_free(output);
            _mm_free(inverse);
            failed++;
            continue;
        }

        printf("  Algorithm: %s\n", fft->lt == 0 ? "Mixed-Radix" : "Bluestein");
        
        // Forward FFT
        fft_exec(fft, input, output);
           printf("boom5");

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
            passed++;
        }
        else
        {
            printf("✗ FAILED (MSE exceeds tolerance)\n");
            failed++;
            
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
    printf("AVX-512 Test Summary: %d passed, %d failed\n", passed, failed);
    printf("===========================================\n\n");

#else
    printf("=== AVX-512 Not Available ===\n");
    printf("Compile with -mavx512f -mavx512dq to enable AVX-512 tests\n\n");
#endif

    //==========================================================================
    // MIXED-RADIX TESTS (Always run these)
    //==========================================================================
    printf("=== Mixed-Radix FFT Tests ===\n\n");
    
    int mixed_radix_lengths[] = {4, 8, 12, 16, 32, 64, 128};
    int num_mixed_tests = sizeof(mixed_radix_lengths) / sizeof(mixed_radix_lengths[0]);
    
    int mixed_passed = 0;
    int mixed_failed = 0;

    for (int test = 0; test < num_mixed_tests; test++)
    {
        int N = mixed_radix_lengths[test];
        printf("Testing Mixed-Radix FFT with N = %d\n", N);

        // Allocate memory
        fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *inverse = (fft_data *)malloc(N * sizeof(fft_data));
        
        if (!input || !output || !inverse)
        {
           // fprintf(stderr, "  ERROR: Memory allocation failed\n");
            if (input) free(input);
            if (output) free(output);
            if (inverse) free(inverse);
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
           // fprintf(stderr, "  ERROR: FFT initialization failed\n");
            if (fft) free_fft(fft);
            if (ifft) free_fft(ifft);
            free(input);
            free(output);
            free(inverse);
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
        free(input);
        free(output);
        free(inverse);
    }
    
    printf("\n===========================================\n");
    printf("Mixed-Radix Test Summary: %d passed, %d failed\n", mixed_passed, mixed_failed);
    printf("===========================================\n\n");

    //==========================================================================
    // BLUESTEIN TESTS (Prime sizes)
    //==========================================================================
    printf("=== Bluestein FFT Tests (Prime Sizes) ===\n\n");
    
    int prime_lengths[] = {7, 11, 13, 17, 19, 23, 29, 31};
    int num_prime_tests = sizeof(prime_lengths) / sizeof(prime_lengths[0]);
    
    int prime_passed = 0;
    int prime_failed = 0;

    for (int test = 0; test < num_prime_tests; test++)
    {
        int N = prime_lengths[test];
        printf("Testing Bluestein FFT with N = %d\n", N);

        fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *inverse = (fft_data *)malloc(N * sizeof(fft_data));
        
        if (!input || !output || !inverse)
        {
        //    fprintf(stderr, "  ERROR: Memory allocation failed\n");
            if (input) free(input);
            if (output) free(output);
            if (inverse) free(inverse);
            prime_failed++;
            continue;
        }

        generate_signal(input, N, 1.0, 1.0);

        fft_object fft = fft_init(N, 1);
        fft_object ifft = fft_init(N, -1);
        
        if (!fft || !ifft)
        {
         //   fprintf(stderr, "  ERROR: FFT initialization failed\n");
            if (fft) free_fft(fft);
            if (ifft) free_fft(ifft);
            free(input);
            free(output);
            free(inverse);
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
        free(input);
        free(output);
        free(inverse);
    }
    
    printf("\n===========================================\n");
    printf("Bluestein Test Summary: %d passed, %d failed\n", prime_passed, prime_failed);
    printf("===========================================\n\n");

    //==========================================================================
    // FINAL SUMMARY
    //==========================================================================
    int total_passed = 0;
    int total_failed = 0;
    
#ifdef HAS_AVX512
    total_passed += passed;
    total_failed += failed;
#endif
    
    total_passed += mixed_passed + prime_passed;
    total_failed += mixed_failed + prime_failed;
    
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