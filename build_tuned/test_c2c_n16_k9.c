/* Standalone: 1D C2C N=16 K=9 with impulse at position [0, 0]. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "planner.h"
#include "env.h"

int main(void) {
    stride_env_init();
    stride_set_num_threads(1);
    stride_registry_t reg; stride_registry_init(&reg);

    int N = 16; size_t K = 9;
    size_t NK = (size_t)N * K;
    double *re = (double *)_aligned_malloc(NK*sizeof(double), 64);
    double *im = (double *)_aligned_malloc(NK*sizeof(double), 64);
    memset(re, 0, NK*sizeof(double));
    memset(im, 0, NK*sizeof(double));
    re[0] = 1.0;  /* impulse at [n=0, k=0] */

    stride_plan_t *plan = stride_auto_plan(N, K, &reg);
    stride_execute_fwd(plan, re, im);

    printf("FFT(impulse at [n=0,k=0]) at [n*K+k] for n=0..15, k=0..2:\n");
    for (int n = 0; n < 16; n++) {
        printf("  n=%-2d ", n);
        for (size_t k = 0; k < 3; k++)
            printf(" (%.2f,%.2f)", re[(size_t)n*K+k], im[(size_t)n*K+k]);
        printf("\n");
    }
    stride_plan_destroy(plan);
    return 0;
}
