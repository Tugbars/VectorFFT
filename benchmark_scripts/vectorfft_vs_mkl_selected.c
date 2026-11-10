/* vectorfft_vs_mkl_selected.c -- VectorFFT against Intel MKL on a selection of commonly used transforms, with the
 * library's estimate plans (VFFT_FLAG_ESTIMATE: nothing measured, so the run takes minutes on any machine).
 * The machinery is vectorfft_vs_mkl.h; the target is benchmark_scripts/CMakeLists.txt's.
 *
 *   cmake --build <build> --target vectorfft_vs_mkl_selected
 *   <build>/benchmark_scripts/vectorfft_vs_mkl_selected [--out DIR]
 *
 * The selection (timed on the i9-14900KF 2026-11-10, gauntlet/results/sortie_*_est_20261110): 1D lengths from
 * 1,920 to 4M points (powers of two and lengths of 2, 3 and 5), the 2D planes of 512 x 512 to 2048 x 2048 and the
 * two video frames, the 3D volumes of 2M to 16M points. */
#include "vectorfft_vs_mkl.h"

static const char *const CELLS[] = {
    /* 1D */
    "1920", "2048", "2560", "3072", "3840", "1048576", "4194304",
    /* 2D: N1 rows of N2 points (1080 x 1920 = a 1920 x 1080 frame) */
    "512x512", "1024x1024", "2048x1024", "2048x2048", "1080x1920", "720x1280",
    /* 3D */
    "128x128x128", "256x128x128", "256x256x64", "64x256x256", "256x256x256",
    NULL
};

int main(int argc, char **argv)
{
    return vs_main(argc, argv, "vectorfft_vs_mkl_selected", "selected transforms, estimate plans", CELLS);
}
