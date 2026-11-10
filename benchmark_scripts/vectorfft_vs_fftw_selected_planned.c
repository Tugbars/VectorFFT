/* vectorfft_vs_fftw_selected_planned.c -- VectorFFT against FFTW on vectorfft_vs_fftw_selected's transforms, with
 * the planner's plans (the library's default planning, which measures, into this machine's wisdom folder
 * src/wisdom/<CPU>/: a transform planned on this machine before is read back at once). FFTW measures its own plans
 * the first time (FFTW_MEASURE) and keeps them for the next runs. vectorfft_launcher.h checks the CPU and runs the
 * benchmark worker; the target is benchmark_scripts/CMakeLists.txt's.
 *
 *   cmake --build <build> --target vectorfft_vs_fftw_selected_planned
 *   <build>/benchmark_scripts/vectorfft_vs_fftw_selected_planned [--cells LIST] [--out DIR] [--avx2 | --avx512] */
#include "vectorfft_launcher.h"

static const vl_program_t P = { "vectorfft_vs_fftw_selected_planned", "selected transforms, the planner's plans",
                                "fftw", "planned", "f64", VL_CELLS_SELECTED };

int main(int argc, char **argv) { return vl_main(argc, argv, &P); }
