/* vectorfft_bench_worker.c -- the benchmark worker of one instruction set: what the vectorfft_vs_* programs run
 * once they have checked the CPU (vectorfft_launcher.h). The machinery is vectorfft_bench.h's; the target is
 * benchmark_scripts/CMakeLists.txt's (one build per instruction set). */
#include "vectorfft_bench.h"

int main(int argc, char **argv)
{
    return vs_main(argc, argv);
}
