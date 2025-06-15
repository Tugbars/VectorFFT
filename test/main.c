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

    printf("=== Complex FFT Tests ===\n");
    for (int test = 0; test < num_complex_tests; test++)
    {
        int N = complex_lengths[test];
        printf("Testing Complex FFT with N = %d\n", N);

        // Allocate memory
        fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *inverse = (fft_data *)malloc(N * sizeof(fft_data));
        if (!input || !output || !inverse)
        {
            fprintf(stderr, "Memory allocation failed for N = %d\n", N);
            free(input);
            free(output);
            free(inverse);
            continue;
        }

        // Generate test signal
        generate_signal(input, N, freq, amplitude);

        // Initialize and perform forward FFT
        fft_object fft = fft_init(N, 1);
        if (!fft)
        {
            fprintf(stderr, "FFT initialization failed for N = %d\n", N);
            free(input);
            free(output);
            free(inverse);
            continue;
        }
        fft_exec(fft, input, output);
        printf("Algorithm: %s\n", fft->lt == 0 ? "Mixed-Radix" : "Bluestein");
        print_complex(output, N, "FFT Output");

        // Initialize and perform inverse FFT
        fft_object ifft = fft_init(N, -1);
        if (!ifft)
        {
            fprintf(stderr, "Inverse FFT initialization failed for N = %d\n", N);
            free_fft(fft);
            free(input);
            free(output);
            free(inverse);
            continue;
        }
        fft_exec(ifft, output, inverse);

        // Scale inverse output
        for (int i = 0; i < N; i++)
        {
            inverse[i].re /= N;
            inverse[i].im /= N;
        }
        print_complex(inverse, N, "Inverse FFT Output");

        // Verify correctness
        double mse = compute_mse(input, inverse, N);
        printf("MSE: %.6e\n", mse);
        if (mse < mse_tolerance)
        {
            printf("Test passed\n");
        }
        else
        {
            printf("Test failed (MSE exceeds tolerance)\n");
        }

        // Cleanup
        free_fft(fft);
        free_fft(ifft);
        free(input);
        free(output);
        free(inverse);
        printf("\n");
    }

    // **Real FFT Tests**
    int real_lengths[] = {4, 8, 16, 32, 64}; // Even lengths only
    int num_real_tests = sizeof(real_lengths) / sizeof(real_lengths[0]);

    printf("=== Real FFT Tests ===\n");
    for (int test = 0; test < num_real_tests; test++)
    {
        int N = real_lengths[test];
        printf("Testing Real FFT with N = %d\n", N);

        // Allocate memory
        fft_type *real_input = (fft_type *)malloc(N * sizeof(fft_type));
        fft_data *complex_output = (fft_data *)malloc(N * sizeof(fft_data)); // Full N for symmetry
        fft_type *real_inverse = (fft_type *)malloc(N * sizeof(fft_type));
        if (!real_input || !complex_output || !real_inverse)
        {
            fprintf(stderr, "Memory allocation failed for N = %d\n", N);
            free(real_input);
            free(complex_output);
            free(real_inverse);
            continue;
        }

        // Generate real-valued test signal
        generate_real_signal(real_input, N, freq, amplitude);
        print_real(real_input, N, "Original Real Signal");

        // Initialize and perform real-to-complex FFT
        fft_real_object r2c = fft_real_init(N, 1);
        if (!r2c)
        {
            fprintf(stderr, "Real FFT initialization failed for N = %d\n", N);
            free(real_input);
            free(complex_output);
            free(real_inverse);
            continue;
        }
        fft_r2c_exec(r2c, real_input, complex_output);
        print_complex(complex_output, N / 2 + 1, "R2C FFT Output"); // Print unique values

        // Initialize and perform complex-to-real FFT
        fft_real_object c2r = fft_real_init(N, -1);
        if (!c2r)
        {
            fprintf(stderr, "Real IFFT initialization failed for N = %d\n", N);
            free_real_fft(r2c);
            free(real_input);
            free(complex_output);
            free(real_inverse);
            continue;
        }
        fft_c2r_exec(c2r, complex_output, real_inverse);

        // Scale inverse output
        for (int i = 0; i < N; i++)
        {
            real_inverse[i] /= N;
        }
        print_real(real_inverse, N, "Reconstructed Real Signal");

        // Verify correctness
        double mse = compute_mse_real(real_input, real_inverse, N);
        printf("MSE: %.6e\n", mse);
        if (mse < mse_tolerance)
        {
            printf("Test passed\n");
        }
        else
        {
            printf("Test failed (MSE exceeds tolerance)\n");
        }

        // Cleanup
        free_real_fft(r2c);
        free_real_fft(c2r);
        free(real_input);
        free(complex_output);
        free(real_inverse);
        printf("\n");
    }

    // **Error Handling Tests**
    printf("=== Error Handling Tests ===\n");

    // Complex FFT: Invalid length
    fft_object bad_fft = fft_init(0, 1);
    if (!bad_fft)
    {
        printf("Complex FFT: Correctly handled invalid length (0)\n");
    }

    // Complex FFT: Invalid direction
    bad_fft = fft_init(8, 0);
    if (!bad_fft)
    {
        printf("Complex FFT: Correctly handled invalid direction (0)\n");
    }

    // Real FFT: Odd length
    fft_real_object bad_r2c = fft_real_init(5, 1);
    if (!bad_r2c)
    {
        printf("Real FFT: Correctly handled odd length (5)\n");
    }

    // Real FFT: Invalid direction
    bad_r2c = fft_real_init(8, 0);
    if (!bad_r2c)
    {
        printf("Real FFT: Correctly handled invalid direction (0)\n");
    }

    // Real FFT: Zero length
    bad_r2c = fft_real_init(0, 1);
    if (!bad_r2c)
    {
        printf("Real FFT: Correctly handled invalid length (0)\n");
    }

    return EXIT_SUCCESS;
}
    

