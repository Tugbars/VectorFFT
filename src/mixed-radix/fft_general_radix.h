#ifndef FFT_GENERAL_RADIX_H
#define FFT_GENERAL_RADIX_H

#include "../fft_plan/fft_planning_types.h"

/**
 * @brief Forward general radix butterfly (arbitrary radix)
 * 
 * Uses precomputed stage twiddles with forward sign: exp(-2πirk/N)
 * Uses precomputed DFT kernel twiddles with forward sign: exp(-2πim/r)
 */
void fft_general_radix_fv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    const fft_data *restrict dft_kernel_tw,  // ⚡ Precomputed!
    int radix,
    int sub_len);

/**
 * @brief Inverse general radix butterfly (arbitrary radix)
 * 
 * Uses precomputed stage twiddles with inverse sign: exp(+2πirk/N)
 * Uses precomputed DFT kernel twiddles with inverse sign: exp(+2πim/r)
 */
void fft_general_radix_bv(
    fft_data *restrict output_buffer,
    const fft_data *restrict sub_outputs,
    const fft_data *restrict stage_tw,
    const fft_data *restrict dft_kernel_tw,  // ⚡ Precomputed!
    int radix,
    int sub_len);

#endif // FFT_GENERAL_RADIX_H