/* vectorfft_vs_mkl_pow2.c -- VectorFFT against Intel MKL on the 1D transform at every power of two from 64 to
 * 4,194,304 (17 lengths), with the library's estimate plans (VFFT_FLAG_ESTIMATE: nothing measured, so the run
 * takes minutes on any machine). The powers of two are the lengths MKL is tuned for most.
 * The machinery is vectorfft_vs_mkl.h's; the target is benchmark_scripts/CMakeLists.txt's.
 *
 *   cmake --build <build> --target vectorfft_vs_mkl_pow2
 *   <build>/benchmark_scripts/vectorfft_vs_mkl_pow2 [--out DIR] */
#include "vectorfft_vs_mkl.h"

static const char *const CELLS[] = {
    "64", "128", "256", "512", "1024", "2048", "4096", "8192", "16384", "32768", "65536",
    "131072", "262144", "524288", "1048576", "2097152", "4194304",
    NULL
};

int main(int argc, char **argv)
{
    return vs_main(argc, argv, "vectorfft_vs_mkl_pow2", "1D powers of two, estimate plans", CELLS, 0);
}
