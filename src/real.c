/**
 * @file real.c
 * @brief Real-to-Complex and Complex-to-Real FFT transformations for real-valued signals.
 * @date March 2, 2025
 * @note Utilizes the high-speed FFT implementation from highspeedFFT.h for efficient transformations.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "real.h"         // Header for real FFT definitions
#include "highspeedFFT.h" // Complex FFT implementation

/**
 * @brief Initializes a real FFT object for real-to-complex transformations.
 *
 * Creates and configures a real FFT object, initializing an underlying complex FFT object
 * for length N/2 and computing twiddle factors for real-valued signal transformations.
 *
 * @param[in] signal_length Length of the input signal (N > 0, must be even).
 * @param[in] transform_direction Direction of the transform (+1 for forward, -1 for inverse).
 * @return fft_real_object Pointer to the initialized real FFT object, or NULL if initialization fails.
 * @note Uses PI2 from highspeedFFT.h for twiddle factor calculations. Sets twiddle2[0].im = 0 explicitly.
 */
fft_real_object fft_real_init(int signal_length, int transform_direction)
{
    if (signal_length <= 0 || signal_length % 2 != 0) {
        fprintf(stderr, "Error: Signal length (%d) must be positive and even\n", signal_length);
        return NULL;
    }

    fft_real_object real_obj = NULL;
    int half_length = signal_length / 2;

    // Allocate memory for the real FFT object, including twiddle factors
    real_obj = (fft_real_object)malloc(sizeof(struct fft_real_set) + sizeof(fft_data) * half_length);
    if (real_obj == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for real FFT object\n");
        return NULL;
    }

    // Initialize the underlying complex FFT object for half the signal length
    real_obj->cobj = fft_init(half_length, transform_direction);
    if (real_obj->cobj == NULL) {
        free(real_obj);
        fprintf(stderr, "Error: Failed to initialize complex FFT object for N/2=%d\n", half_length);
        return NULL;
    }

    // Compute twiddle factors for real-to-complex transformation: e^{-2πi k / N}
    real_obj->twiddle2[0].re = 1.0; // cos(0) for k=0
    real_obj->twiddle2[0].im = 0.0; // sin(0) for k=0 (explicitly zero)
    for (int index = 1; index < half_length; ++index) {
        fft_type angle = PI2 * index / signal_length; // Angle = 2πk/N
        real_obj->twiddle2[index].re = cos(angle);    // cos(2πk/N)
        real_obj->twiddle2[index].im = sin(angle);    // sin(2πk/N)
    }

    return real_obj;
}

/**
 * @brief Performs real-to-complex FFT transformation on real-valued input data.
 *
 * Transforms real-valued input data into complex FFT coefficients using the real FFT object.
 * Packs input as [(x0,x1), (x2,x3), ...] for the complex FFT of length N/2.
 *
 * @param[in] real_obj Real FFT configuration object.
 * @param[in] input_data Real-valued input signal (length N, must be even).
 * @param[out] output_data Complex FFT output (length N/2+1, Hermitian symmetric).
 * @return 0 on success, -1 for invalid inputs, -2 for memory allocation failure.
 * @note Output contains N/2+1 unique bins; X(0) and X(N/2) have zero imaginary parts.
 */
int fft_r2c_exec(fft_real_object real_obj, fft_type *input_data, fft_data *output_data)
{
    if (real_obj == NULL || real_obj->cobj == NULL || input_data == NULL || output_data == NULL) {
        fprintf(stderr, "Error: Invalid real FFT object or data pointers\n");
        return -1;
    }

    int half_length = real_obj->cobj->n_input; // N/2

    // Allocate buffer for complex FFT input/output
    fft_data *buffer = (fft_data *)malloc(sizeof(fft_data) * half_length);
    if (buffer == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for R2C transformation buffer\n");
        return -2;
    }

    // Pack real input into complex pairs: [(x0,x1), (x2,x3), ...]
    for (int index = 0; index < half_length; ++index) {
        buffer[index].re = input_data[2 * index];     // Real part = x[2k]
        buffer[index].im = input_data[2 * index + 1]; // Imag part = x[2k+1]
    }

    // Execute complex FFT on packed data (in-place)
    fft_exec(real_obj->cobj, buffer, buffer);

    // Unpack and combine complex FFT results into real-to-complex output
    output_data[0].re = buffer[0].re + buffer[0].im; // X(0) = F[0].re + F[0].im
    output_data[0].im = 0.0;                         // X(0) imag = 0 (real)

    for (int index = 1; index < half_length; ++index) {
        fft_type temp1 = buffer[index].im + buffer[half_length - index].im; // F[k].im + F[N/2-k].im
        fft_type temp2 = buffer[half_length - index].re - buffer[index].re; // F[N/2-k].re - F[k].re
        // X(k) = (F[k] + F[N/2-k] + e^{-2πi k/N} * (F[k].im + F[N/2-k].im, F[N/2-k].re - F[k].re)) / 2
        output_data[index].re = (buffer[index].re + buffer[half_length - index].re +
                                 (temp1 * real_obj->twiddle2[index].re) + (temp2 * real_obj->twiddle2[index].im)) / 2.0;
        output_data[index].im = (buffer[index].im - buffer[half_length - index].im +
                                 (temp2 * real_obj->twiddle2[index].re) - (temp1 * real_obj->twiddle2[index].im)) / 2.0;
    }

    output_data[half_length].re = buffer[0].re - buffer[0].im; // X(N/2) = F[0].re - F[0].im
    output_data[half_length].im = 0.0;                        // X(N/2) imag = 0 (real)

    free(buffer);
    return 0;
}

/**
 * @brief Performs complex-to-real FFT transformation on complex input data.
 *
 * Transforms complex FFT coefficients back into real-valued output data using the real FFT object.
 * Expects Hermitian symmetric input with N/2+1 unique bins.
 *
 * @param[in] real_obj Real FFT configuration object.
 * @param[in] input_data Complex FFT input (length N/2+1, Hermitian symmetric).
 * @param[out] output_data Real-valued output signal (length N, must be even).
 * @return 0 on success, -1 for invalid inputs, -2 for memory allocation failure.
 * @note Input must be Hermitian symmetric; X(0) and X(N/2) must have zero imaginary parts.
 */
int fft_c2r_exec(fft_real_object real_obj, fft_data *input_data, fft_type *output_data)
{
    if (real_obj == NULL || real_obj->cobj == NULL || input_data == NULL || output_data == NULL) {
        fprintf(stderr, "Error: Invalid real FFT object or data pointers\n");
        return -1;
    }

    int half_length = real_obj->cobj->n_input; // N/2
    int full_length = 2 * half_length;         // N

    // Validate Hermitian symmetry (X(N-k) = conj(X(k)))
    if (fabs(input_data[0].im) > 1e-10 || fabs(input_data[half_length].im) > 1e-10) {
        fprintf(stderr, "Error: Input data at indices 0 and N/2 must have zero imaginary parts\n");
        return -1;
    }
    for (int index = 1; index < half_length; ++index) {
        // Access input_data[full_length - index] assuming input is size N
        if (fabs(input_data[full_length - index].re - input_data[index].re) > 1e-10 ||
            fabs(input_data[full_length - index].im + input_data[index].im) > 1e-10) {
            fprintf(stderr, "Error: Input data is not Hermitian symmetric at index %d\n", index);
            return -1;
        }
    }

    // Allocate buffer for complex FFT input/output
    fft_data *buffer = (fft_data *)malloc(sizeof(fft_data) * half_length);
    if (buffer == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for C2R transformation buffer\n");
        return -2;
    }

    // Pack Hermitian symmetric complex input into complex pairs for FFT
    for (int index = 0; index < half_length; ++index) {
        fft_type temp1 = -input_data[index].im - input_data[half_length - index].im; // -(X[k].im + X[N/2-k].im)
        fft_type temp2 = -input_data[half_length - index].re + input_data[index].re; // -(X[N/2-k].re - X[k].re)
        // F[k] = X[k] + X[N/2-k] + e^{2πi k/N} * (-(X[k].im + X[N/2-k].im), -(X[N/2-k].re - X[k].re))
        buffer[index].re = input_data[index].re + input_data[half_length - index].re +
                           (temp1 * real_obj->twiddle2[index].re) - (temp2 * real_obj->twiddle2[index].im);
        buffer[index].im = input_data[index].im - input_data[half_length - index].im +
                           (temp2 * real_obj->twiddle2[index].re) + (temp1 * real_obj->twiddle2[index].im);
    }

    // Execute complex FFT on packed data (in-place)
    fft_exec(real_obj->cobj, buffer, buffer);

    // Unpack complex FFT results into real output: x[2k] = F[k].re, x[2k+1] = F[k].im
    for (int index = 0; index < half_length; ++index) {
        output_data[2 * index] = buffer[index].re;     // x[2k] = real part
        output_data[2 * index + 1] = buffer[index].im; // x[2k+1] = imag part
    }

    free(buffer);
    return 0;
}

/**
 * @brief Frees a real FFT object and its associated resources.
 *
 * Deallocates memory for the real FFT object and its underlying complex FFT object.
 * Safe to call with NULL or partially initialized objects.
 *
 * @param[in] real_obj Real FFT object to free.
 */
void fft_real_free(fft_real_object real_obj)
{
    if (real_obj != NULL) {
        if (real_obj->cobj != NULL) {
            free_fft(real_obj->cobj);
            real_obj->cobj = NULL; // Prevent double-free
        }
        free(real_obj);
    }
}