/**
 * @file real.h
 * @brief Header for real-to-complex and complex-to-real FFT transformations.
 * @date March 2, 2025
 * @note OPTIMIZED: Added workspace buffer to eliminate malloc/free in hot paths.
 */

#ifndef REAL_H
#define REAL_H

#include "highspeedFFT.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Real FFT object structure.
 * 
 * Contains the underlying complex FFT object, precomputed twiddle factors
 * in Structure-of-Arrays (SoA) format, and a preallocated workspace buffer
 * to eliminate dynamic memory allocation during execution.
 */
struct fft_real_object_s {
    int halfN;              /**< Half the signal length (N/2) */
    fft_object cobj;        /**< Underlying complex FFT object for N/2 */
    double *tw_re;          /**< Real parts of twiddle factors (length N/2) */
    double *tw_im;          /**< Imaginary parts of twiddle factors (length N/2) */
    fft_data *workspace;    /**< Preallocated workspace buffer (length N/2) - NEW */
};

typedef struct fft_real_object_s* fft_real_object;

/**
 * @brief Initializes a real FFT object.
 * 
 * Allocates and initializes a real FFT object for signals of length N (must be even).
 * Preallocates workspace buffer to eliminate malloc/free overhead in exec functions.
 *
 * @param[in] signal_length Length of the input signal (N > 0, must be even).
 * @param[in] transform_direction Direction of the transform (+1 for forward, -1 for inverse).
 * @return fft_real_object Pointer to the initialized object, or NULL on failure.
 */
fft_real_object fft_real_init(int signal_length, int transform_direction);

/**
 * @brief Executes real-to-complex FFT transformation.
 * 
 * Transforms real input x[0..N-1] into complex FFT outputs X[0..N/2] (N even).
 * Uses preallocated workspace buffer - NO dynamic memory allocation.
 *
 * @param[in] real_obj Real FFT configuration object.
 * @param[in] input_data Real-valued input signal (length N).
 * @param[out] output_data Complex FFT output (length N/2+1, Hermitian symmetric).
 * @return 0 on success, -1 on error.
 */
int fft_r2c_exec(fft_real_object real_obj, fft_type *input_data, fft_data *output_data);

/**
 * @brief Executes complex-to-real FFT transformation.
 * 
 * Transforms Hermitian symmetric complex input X[0..N/2] into real output x[0..N-1].
 * Uses preallocated workspace buffer - NO dynamic memory allocation.
 *
 * @param[in] real_obj Real FFT configuration object.
 * @param[in] input_data Complex FFT input (length N/2+1, Hermitian symmetric).
 * @param[out] output_data Real-valued output signal (length N).
 * @return 0 on success, -1 on error.
 */
int fft_c2r_exec(fft_real_object real_obj, fft_data *input_data, fft_type *output_data);

/**
 * @brief Frees a real FFT object.
 * 
 * Deallocates all resources including workspace buffer.
 *
 * @param[in] real_obj Real FFT object to free.
 */
void fft_real_free(fft_real_object real_obj);

#ifdef __cplusplus
}
#endif

#endif /* REAL_H */