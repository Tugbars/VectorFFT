#ifndef FFT_RADIX2_H
#define FFT_RADIX2_H

#include "highspeedFFT.h"

void fft_radix2_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign);

#endif // FFT_RADIX2_H