/* vectorfft_vs_fftw_selected.c -- VectorFFT against FFTW on a selection of commonly used transforms (1D, 2D, 3D),
 * with the library's estimate plans (VFFT_FLAG_ESTIMATE: nothing measured). FFTW is the repository's own build on
 * Windows (gauntlet/fftw/, with its license), the system's libfftw3 elsewhere; it measures its plans the first time
 * (FFTW_MEASURE) and keeps them for the next runs. vectorfft_launcher.h checks the CPU and runs the benchmark
 * worker; the target is benchmark_scripts/CMakeLists.txt's.
 *
 *   cmake --build <build> --target vectorfft_vs_fftw_selected
 *   <build>/benchmark_scripts/vectorfft_vs_fftw_selected [--cells LIST] [--out DIR] [--avx2 | --avx512] */
#include "vectorfft_launcher.h"

static const vl_program_t P = { "vectorfft_vs_fftw_selected", "selected transforms, estimate plans", "fftw", "estimate",
                                "f64", VL_CELLS_SELECTED };

int main(int argc, char **argv) { return vl_main(argc, argv, &P); }
