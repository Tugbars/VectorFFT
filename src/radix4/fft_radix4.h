#ifndef FFT_RADIX4_H
#define FFT_RADIX4_H

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    const double *re;
    const double *im;
} fft_twiddles_soa;

#endif /* FFT_RADIX4_H */
