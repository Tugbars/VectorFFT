/**
 * bench_factorizer.c — Compare heuristic vs exhaustive factorizer
 *
 * For each (N, K) pair, runs:
 *   1. Heuristic factorizer (cache-aware greedy + permutation scoring)
 *   2. Exhaustive search (all factorizations × all orderings, benchmarked)
 *
 * Reports: heuristic pick, exhaustive best, and how close the heuristic gets.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "stride_exhaustive.h"

int main(void) {
    srand(42);
    printf("VectorFFT Factorizer Comparison: Heuristic vs Exhaustive\n");
    printf("=========================================================\n");

    stride_registry_t reg;
    stride_registry_init(&reg);

    stride_cpu_info_t cpu = stride_detect_cpu();
    printf("\nCPU: L1d=%zuKB  L2=%zuKB  line=%zuB\n",
           cpu.l1d_bytes / 1024, cpu.l2_bytes / 1024, cpu.cache_line);

    /* Test cases: mix of pow2, composite, and odd */
    struct { int N; size_t K; } cases[] = {
        /* Small N, various K */
        {60,    4},
        {60,   32},
        {60,  128},

        /* Medium N */
        {200,   4},
        {200,  64},
        {1000,  4},
        {1000, 64},
        {1000, 256},

        /* Pow2 */
        {256,   4},
        {256,  64},
        {1024,  4},
        {1024, 64},
        {4096,  4},
        {4096, 64},

        /* Large composite */
        {2000,  4},
        {2000, 64},
        {5000,  4},
        {5000, 32},

        {0, 0}  /* sentinel */
    };

    for (int i = 0; cases[i].N; i++) {
        stride_compare_strategies(cases[i].N, cases[i].K, &reg);
    }

    printf("\nDone.\n");
    return 0;
}
