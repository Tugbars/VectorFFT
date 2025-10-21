#ifndef FFT_TYPES_H
#define FFT_TYPES_H

// Define fft_type here directly - no dependencies!
#ifndef fft_type
#define fft_type double
#endif

typedef struct fft_t {
    fft_type re;
    fft_type im;
} fft_data;

#endif // FFT_TYPES_H