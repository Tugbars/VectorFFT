#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "real.h" // Provides fft_real_object, fft_r2c_exec, fft_c2r_exec, fft_real_free

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

/**
 * @brief Computes the next power of 2 greater than or equal to n.
 *
 * Used to determine the optimal FFT length for convolution, ensuring efficient
 * power-of-2 FFT computations.
 *
 * @param[in] n Input number.
 * @return int The smallest power of 2 >= n.
 */
int next_power_of_two(int n)
{
    if (n <= 0)
        return 1; // Handle edge case
    return (int)pow(2, ceil(log2(n)));
}

/**
 * @brief Finds the optimal FFT length for convolution.
 *
 * For linear convolution, pads to the next power of 2 >= min_length (e.g., N + L - 1).
 * For circular convolution, pads to the next power of 2 >= max(N, L).
 *
 * @param[in] min_length Minimum required length (e.g., N + L - 1 for linear).
 * @param[in] conv_type Convolution type: "linear" or "circular".
 * @param[in] length1 Length of the first input signal.
 * @param[in] length2 Length of the second input signal.
 * @return int Optimal padded length for FFT, or -1 on error.
 */
int find_optimal_fft_length(int min_length, const char *conv_type, int length1, int length2)
{
    if (strcmp(conv_type, "linear") == 0) {
        return next_power_of_two(min_length);
    } else if (strcmp(conv_type, "circular") == 0) {
        int max_length = MAX(length1, length2);
        return next_power_of_two(max_length);
    } else {
        fprintf(stderr, "Error: Invalid convolution type '%s'. Use 'linear' or 'circular'.\n", conv_type);
        return -1;
    }
}

/**
 * @brief Performs FFT-based convolution of two real-valued signals.
 *
 * Supports linear and circular convolution with output types "full", "same", or "valid".
 * Uses real-to-complex (R2C) and complex-to-real (C2R) FFTs for efficiency, leveraging
 * Hermitian symmetry of real-valued signals.
 *
 * @param[in] type Output type: "full" (full convolution), "same" (central portion matching input1),
 *                 or "valid" (no padding effects). NULL defaults to "full".
 * @param[in] conv_type Convolution type: "linear" (standard) or "circular" (periodic).
 * @param[in] input1 First input signal array (real-valued).
 * @param[in] length1 Length of the first input signal (N > 0).
 * @param[in] input2 Second input signal array (real-valued).
 * @param[in] length2 Length of the second input signal (L > 0).
 * @param[out] output Array to store the convolution result.
 * @return int Length of the output array, or -1 on error.
 *
 * Process:
 * 1. Validate inputs and determine convolution length based on conv_type.
 * 2. Compute optimal padded length for FFT (power of 2).
 * 3. Initialize forward and inverse R2C FFT objects for padded_length.
 * 4. Allocate padded input arrays and complex output arrays (N/2+1 bins).
 * 5. Copy inputs to padded arrays with zero-padding.
 * 6. Perform R2C FFTs on both inputs to obtain N/2+1 complex bins.
 * 7. Compute pointwise product in frequency domain, ensuring Hermitian symmetry.
 * 8. Perform C2R FFT to obtain time-domain convolution.
 * 9. Scale output and extract relevant portion based on type.
 * 10. Clean up memory and return output length.
 *
 * Optimization:
 * - Uses R2C/C2R FFTs to exploit Hermitian symmetry, reducing complex FFT size to N/2.
 * - Allocates only N/2+1 complex bins for R2C outputs and product.
 * - Ensures Hermitian symmetry in complex product for valid C2R input.
 * - Pads to power of 2 for efficient FFT computation.
 * - Minimizes memory usage with targeted allocations.
 *
 * @note For linear convolution, pads to next power of 2 >= N + L - 1.
 *       For circular convolution, pads to next power of 2 >= max(N, L).
 *       Output length depends on type: "full" (N+L-1), "same" (max(N,L)), "valid" (max(N,L)-min(N,L)+1).
 * @warning Assumes input1, input2, and output are non-NULL and properly allocated.
 */
int fft_convolve(const char *type, const char *conv_type, fft_type *input1, int length1,
                 fft_type *input2, int length2, fft_type *output)
{
    // Step 1: Validate inputs
    if (input1 == NULL || input2 == NULL || output == NULL || length1 <= 0 || length2 <= 0) {
        fprintf(stderr, "Error: Invalid inputs for fft_convolve (NULL pointers or non-positive lengths)\n");
        return -1;
    }

    // Step 2: Determine convolution length based on conv_type
    int conv_length;
    if (strcmp(conv_type, "linear") == 0) {
        conv_length = length1 + length2 - 1; // Full linear convolution length
    } else if (strcmp(conv_type, "circular") == 0) {
        conv_length = MAX(length1, length2); // Circular convolution length
    } else {
        fprintf(stderr, "Error: Invalid convolution type '%s'. Use 'linear' or 'circular'.\n", conv_type);
        return -1;
    }

    // Step 3: Find optimal padded length for FFT
    int padded_length = find_optimal_fft_length(conv_length, conv_type, length1, length2);
    if (padded_length == -1) {
        return -1; // Error already printed in find_optimal_fft_length
    }
    int half_padded_length = padded_length / 2 + 1; // Number of complex bins for R2C FFT

    // Step 4: Initialize forward and inverse real FFT objects
    fft_real_object fobj = fft_real_init(padded_length, 1);  // Forward R2C FFT
    fft_real_object iobj = fft_real_init(padded_length, -1); // Inverse C2R FFT
    if (!fobj || !iobj) {
        fprintf(stderr, "Error: Failed to initialize real FFT objects\n");
        fft_real_free(fobj);
        fft_real_free(iobj);
        return -1;
    }

    // Step 5: Allocate memory for padded inputs and FFT outputs
    fft_type *padded_input1 = (fft_type *)calloc(padded_length, sizeof(fft_type)); // Zero-padded
    fft_type *padded_input2 = (fft_type *)calloc(padded_length, sizeof(fft_type)); // Zero-padded
    fft_data *complex_output1 = (fft_data *)malloc(sizeof(fft_data) * half_padded_length); // N/2+1 bins
    fft_data *complex_output2 = (fft_data *)malloc(sizeof(fft_data) * half_padded_length); // N/2+1 bins
    fft_data *complex_product = (fft_data *)malloc(sizeof(fft_data) * padded_length); // Full N for C2R
    fft_type *final_output = (fft_type *)malloc(sizeof(fft_type) * padded_length); // Time-domain result

    if (!padded_input1 || !padded_input2 || !complex_output1 || !complex_output2 || !complex_product || !final_output) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(padded_input1);
        free(padded_input2);
        free(complex_output1);
        free(complex_output2);
        free(complex_product);
        free(final_output);
        fft_real_free(fobj);
        fft_real_free(iobj);
        return -1;
    }

    // Step 6: Copy inputs to padded arrays
    memcpy(padded_input1, input1, length1 * sizeof(fft_type)); // Zero-padded by calloc
    memcpy(padded_input2, input2, length2 * sizeof(fft_type)); // Zero-padded by calloc

    // Step 7: Perform forward real-to-complex FFTs
    if (fft_r2c_exec(fobj, padded_input1, complex_output1) != 0 ||
        fft_r2c_exec(fobj, padded_input2, complex_output2) != 0) {
        fprintf(stderr, "Error: R2C FFT execution failed\n");
        free(padded_input1);
        free(padded_input2);
        free(complex_output1);
        free(complex_output2);
        free(complex_product);
        free(final_output);
        fft_real_free(fobj);
        fft_real_free(iobj);
        return -1;
    }

    // Step 8: Multiply frequency-domain representations pointwise
    // Compute product for 0 to N/2 bins
    complex_product[0].re = complex_output1[0].re * complex_output2[0].re; // X(0) is real
    complex_product[0].im = 0.0; // Ensure zero imag for Hermitian symmetry
    for (int i = 1; i < half_padded_length - 1; i++) {
        complex_product[i].re = complex_output1[i].re * complex_output2[i].re - complex_output1[i].im * complex_output2[i].im;
        complex_product[i].im = complex_output1[i].re * complex_output2[i].im + complex_output1[i].im * complex_output2[i].re;
    }
    complex_product[half_padded_length - 1].re = complex_output1[half_padded_length - 1].re * complex_output2[half_padded_length - 1].re; // X(N/2) is real
    complex_product[half_padded_length - 1].im = 0.0; // Ensure zero imag

    // Mirror to ensure Hermitian symmetry for N/2+1 to N-1
    for (int i = half_padded_length; i < padded_length; i++) {
        int mirror_i = padded_length - i;
        complex_product[i].re = complex_product[mirror_i].re;
        complex_product[i].im = -complex_product[mirror_i].im;
    }

    // Step 9: Perform inverse complex-to-real FFT
    if (fft_c2r_exec(iobj, complex_product, final_output) != 0) {
        fprintf(stderr, "Error: C2R FFT execution failed\n");
        free(padded_input1);
        free(padded_input2);
        free(complex_output1);
        free(complex_output2);
        free(complex_product);
        free(final_output);
        fft_real_free(fobj);
        fft_real_free(iobj);
        return -1;
    }

    // Step 10: Scale the output
    for (int i = 0; i < padded_length; i++) {
        final_output[i] /= padded_length;
    }

    // Step 11: Determine output length and starting point based on type and conv_type
    int start = 0, output_length = 0;
    if (strcmp(conv_type, "linear") == 0) {
        if (type == NULL || strcmp(type, "full") == 0) {
            start = 0;
            output_length = conv_length; // N + L - 1
        } else if (strcmp(type, "same") == 0) {
            int larger_base = MAX(length1, length2);
            start = (conv_length - larger_base) / 2;
            output_length = larger_base;
        } else if (strcmp(type, "valid") == 0) {
            int smaller_length = MIN(length1, length2);
            start = smaller_length - 1;
            output_length = (smaller_length == 0) ? 0 : (MAX(length1, length2) - smaller_length + 1);
        } else {
            fprintf(stderr, "Error: Invalid output type '%s'. Use 'full', 'same', or 'valid'.\n", type);
            output_length = -1;
        }
    } else if (strcmp(conv_type, "circular") == 0) {
        // Circular convolution: output matches the larger input length
        start = 0;
        output_length = MAX(length1, length2);
        // Note: 'same' or 'valid' may need adjustment based on use case
        if (type != NULL && (strcmp(type, "same") == 0 || strcmp(type, "valid") == 0)) {
            fprintf(stderr, "Warning: Output type '%s' may not be well-defined for circular convolution. Using full length.\n", type);
        }
    }

    // Step 12: Copy the relevant portion to output
    if (output_length > 0 && start + output_length <= padded_length) {
        memcpy(output, final_output + start, output_length * sizeof(fft_type));
    } else if (output_length != -1) {
        fprintf(stderr, "Error: Output length %d or start index %d exceeds padded length %d\n",
                output_length, start, padded_length);
        output_length = -1;
    }

    // Step 13: Clean up memory
    free(padded_input1);
    free(padded_input2);
    free(complex_output1);
    free(complex_output2);
    free(complex_product);
    free(final_output);
    fft_real_free(fobj);
    fft_real_free(iobj);

    return output_length;
}