    // SPDX-License-Identifier: BSD-3-Clause
    /**
     * @file real.h
     * @brief Header for Real-to-Complex and Complex-to-Real FFT transformations for real-valued signals.
     * @date March 2, 2025
     * @note Provides functions and data structures for efficient FFT-based transformations of real signals,
     *       utilizing the high-speed FFT implementation from highSpeedFFT.h.
     */

    #ifndef REAL_H_
    #define REAL_H_

    #include "highspeedFFT.h" // Refactored FFT implementation

    #ifdef __cplusplus
    extern "C"
    {
    #endif

        /**
         * @brief Opaque pointer to the real FFT configuration object.
         */
        typedef struct fft_real_set *fft_real_object;

        /**
         * @brief Real FFT configuration structure, containing an underlying complex FFT object and twiddle factors.
         *
         * Stores the configuration for real-to-complex and complex-to-real FFTs, including an underlying complex FFT
         * object for length N/2 transformations and separate arrays for twiddle factor real and imaginary parts in
         * Structure-of-Arrays (SoA) format for SIMD efficiency.
         */
        struct fft_real_set
        {
            fft_object cobj; /**< Underlying complex FFT object for half-length transformations. */
            int halfN;       /**< Half the input signal length (N/2), used for indexing. */
            double* tw_re;   /**< Real parts of twiddle factors e^{-2πi k / N} (length N/2, 32-byte aligned). */
            double* tw_im;   /**< Imaginary parts of twiddle factors e^{-2πi k / N} (length N/2, 32-byte aligned). */
        };
        /**
         * @brief Initializes a real FFT object for real-to-complex transformations.
         *
         * Creates and configures a real FFT object for signals of length N (must be even). Initializes an underlying
         * complex FFT object for length N/2 and precomputes twiddle factors in SoA format (separate real and imaginary
         * arrays) for SIMD efficiency. Twiddle factors are e^{-2πi k / N} for real-to-complex transforms.
         *
         * @param[in] signal_length Length of the input signal (N > 0, must be even).
         * @param[in] transform_direction Direction of the transform (+1 for forward, -1 for inverse).
         * @return fft_real_object Pointer to the initialized real FFT object, or NULL if initialization fails.
         * @warning Exits with an error if signal_length is invalid (non-positive or odd) or memory allocation fails.
         * @note Uses 32-byte aligned memory for twiddle factors to optimize AVX2 performance.
         *       Caller must free the object with fft_real_free().
         */
        fft_real_object fft_real_init(int signal_length, int transform_direction);
        
        /**
         * @brief Performs real-to-complex FFT transformation on real-valued input data.
         *
         * Transforms a real-valued input signal of length N (even) into complex FFT coefficients of length N/2 + 1,
         * leveraging the Hermitian symmetry of the output. Uses a complex FFT of length N/2 and vectorized operations
         * with AVX2 if available.
         *
         * @param[in] real_obj Real FFT configuration object.
         * @param[in] input_data Real-valued input signal (length N, must be even).
         * @param[out] output_data Complex FFT output (length N/2 + 1, Hermitian symmetric).
         * @return 0 on success, -1 for invalid inputs, -2 for memory allocation failure.
         * @warning Exits with an error if real_obj, input_data, or output_data is NULL, or memory allocation fails.
         * @note Output is Hermitian symmetric (X(N-k) = conj(X(k))), with X(0) and X(N/2) having zero imaginary parts.
         */
        int fft_r2c_exec(fft_real_object real_obj, fft_type *input_data, fft_data *output_data);
        
        /**
         * @brief Performs complex-to-real FFT transformation on complex input data.
         *
         * Transforms Hermitian symmetric complex FFT coefficients of length N/2 + 1 into a real-valued output signal
         * of length N (even). Uses a complex IFFT of length N/2 and vectorized operations with AVX2 if available.
         *
         * @param[in] real_obj Real FFT configuration object.
         * @param[in] input_data Complex FFT input (length N/2 + 1, Hermitian symmetric).
         * @param[out] output_data Real-valued output signal (length N, must be even).
         * @return 0 on success, -1 for invalid inputs, -2 for memory allocation failure.
         * @warning Exits with an error if real_obj, input_data, or output_data is NULL, or memory allocation fails.
         * @note Input must be Hermitian symmetric (X(N-k) = conj(X(k))), with X(0) and X(N/2) having zero imaginary parts.
         */
        int fft_c2r_exec(fft_real_object real_obj, fft_data *input_data, fft_type *output_data);
        
        /**
         * @brief Frees the memory allocated for a real FFT object.
         *
         * Releases the dynamically allocated memory for the real FFT object, its underlying complex FFT object,
         * and twiddle factor arrays. Safe to call with NULL or partially initialized objects.
         *
         * @param[in] real_obj Real FFT object to free (may be NULL).
         */
        void fft_real_free(fft_real_object real_obj);
    #ifdef __cplusplus
    }
    #endif

    #endif /* REAL_H_ */
