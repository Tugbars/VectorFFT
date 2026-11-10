/* vectorfft_vs_mkl_selected_planned.c -- VectorFFT against Intel MKL on vectorfft_vs_mkl_selected's transforms,
 * with the planner's plans (the library's default planning, which measures). Each transform is planned first into
 * this machine's wisdom folder (src/wisdom/<CPU>/): one this machine planned before is read back at once, a new one
 * is measured and saved there, so the first run on a new machine takes longer and the next ones do not.
 * vectorfft_launcher.h checks the CPU and runs the benchmark worker; the target is benchmark_scripts/CMakeLists.txt's.
 *
 *   cmake --build <build> --target vectorfft_vs_mkl_selected_planned
 *   <build>/benchmark_scripts/vectorfft_vs_mkl_selected_planned [--cells LIST] [--out DIR] [--avx2 | --avx512] */
#include "vectorfft_launcher.h"

static const vl_program_t P = { "vectorfft_vs_mkl_selected_planned", "selected transforms, the planner's plans", "mkl",
                                "planned", "f64", VL_CELLS_SELECTED };

int main(int argc, char **argv) { return vl_main(argc, argv, &P); }
