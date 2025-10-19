#ifndef FFT_GENERAL_RADIX_H
#define FFT_GENERAL_RADIX_H

#include "fft_planning_types.h"

/**
 * @brief Forward general radix butterfly (arbitrary radix)
 * 
 * Uses precomputed stage twiddles with forward sign: exp(-2πirk/N)
 * Computes DFT kernel twiddles on-the-fly with forward sign
 */
void fft_general_radix_fv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    int radix,
    int sub_len);

/**
 * @brief Inverse general radix butterfly (arbitrary radix)
 * 
 * Uses precomputed stage twiddles with inverse sign: exp(+2πirk/N)
 * Computes DFT kernel twiddles on-the-fly with inverse sign
 */
void fft_general_radix_bv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    int radix,
    int sub_len);

#endif // FFT_GENERAL_RADIX_H