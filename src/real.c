/**
 * @file real.c
 * @brief Real-to-Complex and Complex-to-Real FFT transformations for real-valued signals.
 * @date March 2, 2025
 * @note Utilizes the high-speed FFT implementation from highSpeedFFT.h for efficient transformations.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "real.h"         // Assumed header with fft_real_object, fft_data, fft_type definitions
#include "highspeedFFT.h" // Refactored FFT implementation

/**
 * @brief Initializes a real FFT object for real-to-complex transformations.
 *
 * Creates and configures a real FFT object, initializing an underlying complex FFT object
 * and computing twiddle factors for real-valued signal transformations.
 *
 * @param[in] signal_length Length of the input signal (N > 0, must be even).
 * @param[in] transform_direction Direction of the transform (sgn = +1 for forward, -1 for inverse).
 * @return fft_real_object Pointer to the initialized real FFT object, or NULL if initialization fails.
 * @warning If memory allocation fails or N is invalid/odd, the function exits with an error.
 * @note Uses PI = 3.1415926535897932384626433832795 for twiddle factor calculations.
 */
fft_real_object fft_real_init(int signal_length, int transform_direction)
{
    if (signal_length <= 0 || signal_length % 2 != 0)
    {
        fprintf(stderr, "Error: Signal length (%d) must be positive and even\n", signal_length);
        exit(EXIT_FAILURE);
    }

    fft_real_object real_obj = NULL;
    const fft_type PI = 3.1415926535897932384626433832795;
    int half_length = signal_length / 2, index;

    // Allocate memory for the real FFT object, including space for twiddle factors
    real_obj = (fft_real_object)malloc(sizeof(struct fft_real_set) + sizeof(fft_data) * half_length);
    if (real_obj == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for real FFT object\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the underlying complex FFT object for half the signal length
    real_obj->cobj = fft_init(half_length, transform_direction);
    if (real_obj->cobj == NULL)
    {
        free(real_obj);
        fprintf(stderr, "Error: Failed to initialize complex FFT object\n");
        exit(EXIT_FAILURE);
    }

    // Compute twiddle factors for real-to-complex transformation
    for (index = 0; index < half_length; ++index)
    {
        fft_type angle = PI2 * index / signal_length;
        real_obj->twiddle2[index].re = cos(angle);
        real_obj->twiddle2[index].im = sin(angle);
    }

    return real_obj;
}

/**
 * @brief Performs real-to-complex FFT transformation on real-valued input data.
 *
 * Transforms real-valued input data into complex FFT coefficients using the real FFT object,
 * leveraging the complex FFT implementation from highSpeedFFT.h.
 *
 * @param[in] real_obj Real FFT configuration object.
 * @param[in] input_data Real-valued input signal (length N, must be even).
 * @param[out] output_data Complex FFT output (length N/2 + 1, Hermitian symmetric).
 * @warning If memory allocation fails, or inputs are invalid, the function exits with an error.
 * @note Output is Hermitian symmetric, with real_obj->N/2 + 1 unique complex numbers.
 */
void fft_r2c_exec(fft_real_object real_obj, fft_type *input_data, fft_data *output_data)
{
    if (real_obj == NULL || input_data == NULL || output_data == NULL)
    {
        fprintf(stderr, "Error: Invalid real FFT object or data pointers\n");
        exit(EXIT_FAILURE);
    }

    int half_length = real_obj->cobj->N, full_length = 2 * half_length, index;
    fft_data *complex_input = (fft_data *)malloc(sizeof(fft_data) * half_length);
    fft_data *complex_output = (fft_data *)malloc(sizeof(fft_data) * half_length);

    if (complex_input == NULL || complex_output == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for R2C transformation buffers\n");
        free(complex_input);
        free(complex_output);
        exit(EXIT_FAILURE);
    }

    // Pack real input into complex pairs (real, imag)
    for (index = 0; index < half_length; ++index)
    {
        complex_input[index].re = input_data[2 * index];
        complex_input[index].im = input_data[2 * index + 1];
    }

    // Execute complex FFT on packed data
    fft_exec(real_obj->cobj, complex_input, complex_output);

    // Unpack and combine complex FFT results into real-to-complex output
    output_data[0].re = complex_output[0].re + complex_output[0].im;
    output_data[0].im = 0.0;

    for (index = 1; index < half_length; ++index)
    {
        fft_type temp1 = complex_output[index].im + complex_output[half_length - index].im;
        fft_type temp2 = complex_output[half_length - index].re - complex_output[index].re;
        output_data[index].re = (complex_output[index].re + complex_output[half_length - index].re +
                                 (temp1 * real_obj->twiddle2[index].re) + (temp2 * real_obj->twiddle2[index].im)) /
                                2.0;
        output_data[index].im = (complex_output[index].im - complex_output[half_length - index].im +
                                 (temp2 * real_obj->twiddle2[index].re) - (temp1 * real_obj->twiddle2[index].im)) /
                                2.0;
    }

    output_data[half_length].re = complex_output[0].re - complex_output[0].im;
    output_data[half_length].im = 0.0;

    // Ensure Hermitian symmetry (copy and negate imaginary parts for negative frequencies)
    for (index = 1; index < half_length; ++index)
    {
        output_data[full_length - index].re = output_data[index].re;
        output_data[full_length - index].im = -output_data[index].im;
    }

    free(complex_input);
    free(complex_output);
}

/**
 * @brief Performs complex-to-real FFT transformation on complex input data.
 *
 * Transforms complex FFT coefficients back into real-valued output data using the real FFT object,
 * leveraging the complex FFT implementation from highSpeedFFT.h.
 *
 * @param[in] real_obj Real FFT configuration object.
 * @param[in] input_data Complex FFT input (length N/2 + 1, Hermitian symmetric).
 * @param[out] output_data Real-valued output signal (length N, must be even).
 * @warning If memory allocation fails, or inputs are invalid, the function exits with an error.
 * @note Assumes input_data is Hermitian symmetric, with real_obj->N/2 + 1 unique complex numbers.
 */
void fft_c2r_exec(fft_real_object real_obj, fft_data *input_data, fft_type *output_data)
{
    if (real_obj == NULL || input_data == NULL || output_data == NULL)
    {
        fprintf(stderr, "Error: Invalid real FFT object or data pointers\n");
        exit(EXIT_FAILURE);
    }

    int half_length = real_obj->cobj->N, full_length = 2 * half_length, index;
    fft_data *complex_input = (fft_data *)malloc(sizeof(fft_data) * half_length);
    fft_data *complex_output = (fft_data *)malloc(sizeof(fft_data) * half_length);

    if (complex_input == NULL || complex_output == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for C2R transformation buffers\n");
        free(complex_input);
        free(complex_output);
        exit(EXIT_FAILURE);
    }

    // Pack Hermitian symmetric complex input into complex pairs for FFT
    for (index = 0; index < half_length; ++index)
    {
        fft_type temp1 = -input_data[index].im - input_data[half_length - index].im;
        fft_type temp2 = -input_data[half_length - index].re + input_data[index].re;
        complex_input[index].re = input_data[index].re + input_data[half_length - index].re +
                                  (temp1 * real_obj->twiddle2[index].re) - (temp2 * real_obj->twiddle2[index].im);
        complex_input[index].im = input_data[index].im - input_data[half_length - index].im +
                                  (temp2 * real_obj->twiddle2[index].re) + (temp1 * real_obj->twiddle2[index].im);
    }

    // Execute complex FFT on packed data
    fft_exec(real_obj->cobj, complex_input, complex_output);

    // Unpack complex FFT results into real output
    for (index = 0; index < half_length; ++index)
    {
        output_data[2 * index] = complex_output[index].re;
        output_data[2 * index + 1] = complex_output[index].im;
    }

    free(complex_input);
    free(complex_output);
}



/*
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
    

*/