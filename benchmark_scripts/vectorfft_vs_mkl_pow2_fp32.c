/* vectorfft_vs_mkl_pow2_fp32.c -- VectorFFT against Intel MKL in single precision (FP32) on the 1D transform at
 * every power of two from 64 to 4,194,304 (17 lengths), with the planner's plans: the FP32 library has no estimate,
 * so each length is planned first into this machine's wisdom folder (src/wisdom/<CPU>/; a length planned on this
 * machine before is read back at once). Built with -DVFFT_ENABLE_FLOAT=ON. vectorfft_launcher.h checks the CPU and
 * runs the benchmark worker; the target is benchmark_scripts/CMakeLists.txt's.
 *
 *   cmake --build <build> --target vectorfft_vs_mkl_pow2_fp32
 *   <build>/benchmark_scripts/vectorfft_vs_mkl_pow2_fp32 [--cells LIST] [--out DIR] [--avx2 | --avx512] */
#include "vectorfft_launcher.h"

static const vl_program_t P = { "vectorfft_vs_mkl_pow2_fp32", "1D powers of two, single precision, the planner's plans",
                                "mkl", "planned", "f32", VL_CELLS_POW2 };

int main(int argc, char **argv) { return vl_main(argc, argv, &P); }
