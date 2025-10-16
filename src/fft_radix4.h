#ifndef FFT_RADIX4_H
#define FFT_RADIX4_H

#include "highspeedFFT.h"

void fft_radix4_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign);

#endif // FFT_RADIX3_H