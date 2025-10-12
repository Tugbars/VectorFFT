#include "highspeedFFT.h"
#include "real.h"
#include <math.h>

#define M_PI 3.14159265358979323846

// Utility functions from highSpeedFFT.c
void generate_signal(fft_data *signal, int length, double freq, double amplitude)
{
    for (int i = 0; i < length; i++)
    {
        signal[i].re = amplitude * sin(2.0 * M_PI * freq * i / length);
        signal[i].im = 0.0; // Real-valued input
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
    for (int i = 0; i < length; i++)
    {
        printf("  [%d] %.6f + %.6fi\n", i, data[i].re, data[i].im);
    }
    printf("\n");
}

// Utility functions from real.c
void generate_real_signal(fft_type *signal, int length, double freq, double amplitude)
{
    for (int i = 0; i < length; i++)
    {
        signal[i] = amplitude * sin(2.0 * M_PI * freq * i / length);
    }
}

double compute_mse_real(fft_type *original, fft_type *reconstructed, int length)
{
    double mse = 0.0;
    for (int i = 0; i < length; i++)
    {
        double diff = original[i] - reconstructed[i];
        mse += diff * diff;
    }
    return mse / length;
}

void print_real(fft_type *data, int length, const char *label)
{
    printf("%s:\n", label);
    for (int i = 0; i < length; i++)
    {
        printf("  [%d] %.6f\n", i, data[i]);
    }
    printf("\n");
}

void free_real_fft(fft_real_object real_obj)
{
    // Code to free the real FFT object, e.g., freeing memory allocated for real_obj
    if (real_obj != NULL)
    {
        free(real_obj->cobj); // Example: freeing the underlying complex FFT object
        free(real_obj);       // Freeing the real FFT object itself
    }
}

int main()
{
    // Common parameters
    const double freq = 2.0;            // Frequency of test signal
    const double amplitude = 1.0;       // Amplitude of test signal
    const double mse_tolerance = 1e-10; // MSE tolerance for passing tests

    // **Complex FFT Tests**
    int complex_lengths[] = {4, 8, 15, 20, 64}; // Mixed-radix: 4, 8, 64; Bluestein: 15, 20
    int num_complex_tests = sizeof(complex_lengths) / sizeof(complex_lengths[0]);

    // **AVX-512 Radix-2 Specific Tests**
#ifdef HAS_AVX512
    printf("=== AVX-512 Radix-2 Tests ===\n");
    printf("AVX-512 support detected!\n\n");
    
    // Test pure power-of-2 sizes that will use radix-2
    int radix2_lengths[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    int num_radix2_tests = sizeof(radix2_lengths) / sizeof(radix2_lengths[0]);

    for (int test = 0; test < num_radix2_tests; test++)
    {
        int N = radix2_lengths[test];
        printf("Testing AVX-512 Radix-2 FFT with N = %d\n", N);

        // Allocate aligned memory for AVX-512 (64-byte alignment)
        fft_data *input = (fft_data *)_mm_malloc(N * sizeof(fft_data), 64);
        fft_data *output = (fft_data *)_mm_malloc(N * sizeof(fft_data), 64);
        fft_data *inverse = (fft_data *)_mm_malloc(N * sizeof(fft_data), 64);
        
        if (!input || !output || !inverse)
        {
            fprintf(stderr, "Aligned memory allocation failed for N = %d\n", N);
            _mm_free(input);
            _mm_free(output);
            _mm_free(inverse);
            continue;
        }

        // Generate test signal (multiple frequencies for better testing)
        for (int i = 0; i < N; i++)
        {
            input[i].re = sin(2.0 * M_PI * 5.0 * i / N) + 
                          0.5 * cos(2.0 * M_PI * 13.0 * i / N);
            input[i].im = 0.0;
        }

        // Forward FFT
        fft_object fft = fft_init(N, 1);
        if (!fft)
        {
            fprintf(stderr, "FFT initialization failed for N = %d\n", N);
            _mm_free(input);
            _mm_free(output);
            _mm_free(inverse);
            continue;
        }
        
        // Verify it's using the right algorithm
        printf("  Algorithm: %s\n", fft->lt == 0 ? "Mixed-Radix (will use radix-2)" : "Bluestein");
        
        fft_exec(fft, input, output);

        // Inverse FFT
        fft_object ifft = fft_init(N, -1);
        if (!ifft)
        {
            fprintf(stderr, "Inverse FFT initialization failed for N = %d\n", N);
            free_fft(fft);
            _mm_free(input);
            _mm_free(output);
            _mm_free(inverse);
            continue;
        }
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
        }
        else
        {
            printf("✗ FAILED (MSE exceeds tolerance)\n");
            // Print first few values for debugging
            printf("  First 4 values:\n");
            for (int i = 0; i < 4 && i < N; i++)
            {
                printf("    Input[%d]:  %.6f + %.6fi\n", i, input[i].re, input[i].im);
                printf("    Output[%d]: %.6f + %.6fi\n", i, inverse[i].re, inverse[i].im);
            }
        }

        // Cleanup
        free_fft(fft);
        free_fft(ifft);
        _mm_free(input);
        _mm_free(output);
        _mm_free(inverse);
    }
    printf("\n");
#else
    printf("=== AVX-512 Not Available ===\n");
    printf("Compile with -mavx512f -mavx512dq to enable AVX-512 tests\n\n");
#endif

    return EXIT_SUCCESS;
}
    

