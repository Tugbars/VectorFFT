/* vectorfft_vs_mkl_selected.c -- VectorFFT against Intel MKL on a selection of commonly used transforms (1D, 2D,
 * 3D), with the library's estimate plans (VFFT_FLAG_ESTIMATE: nothing measured, so the run takes minutes on any
 * machine). vectorfft_launcher.h checks the CPU and runs the benchmark worker; the target is
 * benchmark_scripts/CMakeLists.txt's.
 *
 *   cmake --build <build> --target vectorfft_vs_mkl_selected
 *   <build>/benchmark_scripts/vectorfft_vs_mkl_selected [--cells LIST] [--out DIR] [--avx2 | --avx512] */
#include "vectorfft_launcher.h"

static const vl_program_t P = { "vectorfft_vs_mkl_selected", "selected transforms, estimate plans", "mkl", "estimate",
                                "f64", VL_CELLS_SELECTED };

int main(int argc, char **argv) { return vl_main(argc, argv, &P); }
