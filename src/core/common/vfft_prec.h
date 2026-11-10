/* vfft_prec.h -- the element precision of this translation unit.
 *
 * The core is one translation unit (vfft.c). The FP32 library is the same source compiled a
 * second time with VFFT_FLOAT defined (vfftf.c): vfft_real is then float, and the kernels,
 * registries, twiddle tables and buffers of the served path are float. Everything that is not
 * sample data stays double at both precisions: timings, tolerances, the cos/sin computation
 * of every table (computed in double or long double, rounded once at the store).
 *
 * At FP32 the served set is 1D C2C, interleaved, power-of-two N (include/vfftf.h); the engines
 * outside it are not compiled into the FP32 unit. */
#ifndef VFFT_PREC_H
#define VFFT_PREC_H

#ifdef VFFT_FLOAT
typedef float vfft_real;
#define VFFT_PREC_TAG "f32"
#else
typedef double vfft_real;
#define VFFT_PREC_TAG "f64"
#endif

#endif /* VFFT_PREC_H */
