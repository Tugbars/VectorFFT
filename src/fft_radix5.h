#ifndef FFT_RADIX5_H
#define FFT_RADIX5_H

#include "highspeedFFT.h"

void fft_radix5_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign);

#endif // FFT_RADIX3_H