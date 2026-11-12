/* vectorfft_vs_fftw_pow2.c -- VectorFFT against FFTW on the 1D transform at every power of two from 64 to
 * 4,194,304 (17 lengths), with the library's estimate plans (VFFT_FLAG_ESTIMATE: nothing measured). FFTW measures
 * its own plans the first time (FFTW_MEASURE) and keeps them for the next runs. vectorfft_launcher.h checks the CPU
 * and runs the benchmark worker; the target is benchmark_scripts/CMakeLists.txt's.
 *
 *   cmake --build <build> --target vectorfft_vs_fftw_pow2
 *   <build>/benchmark_scripts/vectorfft_vs_fftw_pow2 [--cells LIST] [--out DIR] [--inplace] [--avx2 | --avx512] */
#include "vectorfft_launcher.h"

static const vl_program_t P = { "vectorfft_vs_fftw_pow2", "1D powers of two, estimate plans", "fftw", "estimate", "f64",
                                VL_CELLS_POW2 };

int main(int argc, char **argv) { return vl_main(argc, argv, &P); }
