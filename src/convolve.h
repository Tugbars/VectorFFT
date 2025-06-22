#ifndef CONVOLVE_H
#define CONVOLVE_H

#include "real.h" // Provides fft_type, fft_data, fft_real_object, fft_r2c_exec, fft_c2r_exec

/**
 * @file convolve.h
 * @brief FFT-based convolution for real-valued signals.
 * @date June 22, 2025
 * @note Utilizes real-to-complex and complex-to-real FFT transformations from real.h
 *       for efficient convolution of real-valued signals.
 */

/**
 * @brief Computes the next power of 2 greater than or equal to n.
 *
 * Used to determine the optimal FFT length for convolution, ensuring efficient
 * power-of-2 FFT computations.
 *
 * @param[in] n Input number.
 * @return int The smallest power of 2 >= n.
 */
int next_power_of_two(int n);

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
int find_optimal_fft_length(int min_length, const char *conv_type, int length1, int length2);

/**
 * @brief Performs FFT-based convolution of two real-valued signals.
 *
 * Supports linear and circular convolution with output types "full", "same", or "valid".
 * Uses real-to-complex (R2C) and complex-to-real (C2R) FFTs for efficiency, leveraging
 * Hermitian symmetry of real-valued signals.
 *
 * @param[in] type Output type: "full" (full convolution, N+L-1 points),
 *                 "same" (central portion matching larger input, max(N,L) points),
 *                 or "valid" (no padding effects, max(N,L)-min(N,L)+1 points).
 *                 NULL defaults to "full".
 * @param[in] conv_type Convolution type: "linear" (standard) or "circular" (periodic).
 * @param[in] input1 First input signal array (real-valued, length N).
 * @param[in] length1 Length of the first input signal (N > 0).
 * @param[in] input2 Second input signal array (real-valued, length L).
 * @param[in] length2 Length of the second input signal (L > 0).
 * @param[out] output Array to store the convolution result.
 * @return int Length of the output array, or -1 on error (invalid inputs, memory failure, or invalid type).
 */
int fft_convolve(const char *type, const char *conv_type, fft_type *input1, int length1,
                 fft_type *input2, int length2, fft_type *output);

#endif // CONVOLVE_H