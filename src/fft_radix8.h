#ifndef FFT_RADIX8_H
#define FFT_RADIX8_H

#include "highspeedFFT.h"

void fft_radix8_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign);

#endif // FFT_RADIX3_H