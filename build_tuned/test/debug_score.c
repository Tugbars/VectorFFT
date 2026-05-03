/* debug_score.c — dump stride_score_factorization for hand-picked factorizations
 * to see why estimate is mis-picking on power-of-two N.
 *
 * Build:
 *   python build.py --src test/debug_score.c --vfft
 */
#include <stdio.h>
#include "vfft.h"
#include "factorizer.h"

static void score_one(int N, size_t K, const int *factors, int nf,
                      const stride_cpu_info_t *cpu)
{
    double sc = stride_score_factorization(factors, nf, K, N, cpu);
    printf("  N=%-5d K=%-5zu  ", N, K);
    for (int i = 0; i < nf; i++) printf("%s%d", i ? "x" : "", factors[i]);
    printf("    score = %.0f\n", sc);
}

int main(void) {
    vfft_init();
    stride_cpu_info_t cpu = stride_detect_cpu();
    printf("L1=%zu KB  L2=%zu KB\n",
           cpu.l1d_bytes / 1024, cpu.l2_bytes / 1024);

    /* N=8 K=256 candidate factorizations */
    printf("\n[ N=8 K=256 ]\n");
    score_one(8, 256, (int[]){8},        1, &cpu);
    score_one(8, 256, (int[]){4, 2},     2, &cpu);
    score_one(8, 256, (int[]){2, 4},     2, &cpu);
    score_one(8, 256, (int[]){2, 2, 2},  3, &cpu);

    /* N=64 K=256 */
    printf("\n[ N=64 K=256 ]\n");
    score_one(64, 256, (int[]){8, 8},                2, &cpu);
    score_one(64, 256, (int[]){4, 4, 4},             3, &cpu);
    score_one(64, 256, (int[]){4, 4, 2, 2},          4, &cpu);
    score_one(64, 256, (int[]){4, 2, 2, 2, 2},       5, &cpu);
    score_one(64, 256, (int[]){2, 2, 2, 2, 2, 2},    6, &cpu);
    score_one(64, 256, (int[]){32, 2},               2, &cpu);
    score_one(64, 256, (int[]){2, 32},               2, &cpu);

    /* N=64 K=1024 */
    printf("\n[ N=64 K=1024 ]\n");
    score_one(64, 1024, (int[]){8, 8},               2, &cpu);
    score_one(64, 1024, (int[]){4, 4, 4},            3, &cpu);
    score_one(64, 1024, (int[]){32, 2},              2, &cpu);

    return 0;
}
