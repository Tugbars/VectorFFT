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
         */
        struct fft_real_set
        {
            fft_object cobj;      /**< Underlying complex FFT object for half-length transformations. */
            fft_data twiddle2[]; /**< Flexible array for precomputed twiddle factors used in real-to-complex transformations. */
        };

        /**
         * @brief Initializes a real FFT object for real-to-complex transformations.
         *
         * Creates and configures a real FFT object, initializing an underlying complex FFT object
         * and computing twiddle factors for real-valued signal transformations.
         *
         * @param[in] signal_length Length of the input signal (N > 0, must be even).
         * @param[in] transform_direction Direction of the transform (+1 for forward, -1 for inverse).
         * @return fft_real_object Pointer to the initialized real FFT object, or NULL if initialization fails.
         * @warning If memory allocation fails or N is invalid/odd, the function will exit with an error.
         * @note Uses PI2 from highSpeedFFT.h for twiddle factor calculations.
         */
        fft_real_object fft_real_init(int signal_length, int transform_direction);

        /**
         * @brief Performs real-to-complex FFT transformation on real-valued input data.
         *
         * Transforms real-valued input data into complex FFT coefficients using the real FFT object,
         * leveraging the complex FFT implementation from highSpeedFFT.h.
         *
         * @param[in] real_obj Real FFT configuration object.
         * @param[in] input_data Real-valued input signal (length N, must be even).
         * @param[out] output_data Complex FFT output (length N/2 + 1, Hermitian symmetric).
         * @warning If the real FFT object or data pointers are invalid, or memory allocation fails, the function exits with an error.
         * @note Output is Hermitian symmetric, with N/2 + 1 unique complex numbers.
         */
        int fft_r2c_exec( fft_real_object real_obj,
                  fft_type *input_data,
                  fft_data  *output_data );

        /**
         * @brief Performs complex-to-real FFT transformation on complex input data.
         *
         * Transforms complex FFT coefficients back into real-valued output data using the real FFT object,
         * leveraging the complex FFT implementation from highSpeedFFT.h.
         *
         * @param[in] real_obj Real FFT configuration object.
         * @param[in] input_data Complex FFT input (length N/2 + 1, Hermitian symmetric).
         * @param[out] output_data Real-valued output signal (length N, must be even).
         * @warning If the real FFT object or data pointers are invalid, or memory allocation fails, the function exits with an error.
         * @note Assumes input_data is Hermitian symmetric, with N/2 + 1 unique complex numbers.
         */
        int fft_c2r_exec(fft_real_object real_obj, fft_data *input_data, fft_type *output_data);

        /**
         * @brief Frees the memory allocated for a real FFT object.
         *
         * Releases the dynamically allocated memory for the real FFT object and its underlying complex FFT object.
         *
         * @param[in] real_obj Real FFT object to free (may be NULL).
         * @note Safely handles NULL pointers to prevent crashes.
         */
        void fft_real_free(fft_real_object real_obj);

    #ifdef __cplusplus
    }
    #endif

    #endif /* REAL_H_ */