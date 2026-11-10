/* vectorfft_vs_mkl_selected_planned.c -- VectorFFT against Intel MKL on vectorfft_vs_mkl_selected's transforms,
 * with the planner's plans (the library's default planning, which measures). Each transform is planned first into
 * this machine's wisdom folder (src/wisdom/<CPU>/): one this machine planned before is read back at once, a new one
 * is measured and saved there, so the first run on a new machine takes longer and the next ones do not.
 * The selection (VS_SELECTED) and the machinery are vectorfft_vs_mkl.h's; the target is
 * benchmark_scripts/CMakeLists.txt's.
 *
 *   cmake --build <build> --target vectorfft_vs_mkl_selected_planned
 *   <build>/benchmark_scripts/vectorfft_vs_mkl_selected_planned [--out DIR] */
#include "vectorfft_vs_mkl.h"

int main(int argc, char **argv)
{
    return vs_main(argc, argv, "vectorfft_vs_mkl_selected_planned", "selected transforms, the planner's plans",
                   VS_SELECTED, 1);
}
