/* vectorfft_vs_mkl_selected.c -- VectorFFT against Intel MKL on a selection of commonly used transforms, with the
 * library's estimate plans (VFFT_FLAG_ESTIMATE: nothing measured, so the run takes minutes on any machine).
 * The selection (VS_SELECTED) and the machinery are vectorfft_vs_mkl.h's; the target is
 * benchmark_scripts/CMakeLists.txt's.
 *
 *   cmake --build <build> --target vectorfft_vs_mkl_selected
 *   <build>/benchmark_scripts/vectorfft_vs_mkl_selected [--out DIR]
 *
 * vectorfft_vs_mkl_selected_planned times the same transforms with the planner's plans. */
#include "vectorfft_vs_mkl.h"

int main(int argc, char **argv)
{
    return vs_main(argc, argv, "vectorfft_vs_mkl_selected", "selected transforms, estimate plans", VS_SELECTED, 0);
}
