#ifndef FFT_RADIX16_H
#define FFT_RADIX16_H

#include "highspeedFFT.h"

void fft_radix16_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign);

#endif // FFT_RADIX3_H