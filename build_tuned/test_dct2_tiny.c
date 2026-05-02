/* test_dct2_tiny.c — minimum reproducer for DCT-II */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "planner.h"
#include "dct.h"
#include "env.h"

int main(void) {
    stride_env_init();
    stride_set_num_threads(1);
    stride_registry_t reg;
    stride_registry_init(&reg);

    int N = 8;
    size_t K = 4;     /* avoid K=1 SIMD edge case for now */
    double *in  = (double *)_aligned_malloc(N * K * sizeof(double), 64);
    double *out = (double *)_aligned_malloc(N * K * sizeof(double), 64);
    /* Layout: in[n*K + k]. Set K-column 0 to [1..8]; other columns can be 0. */
    for (size_t i = 0; i < (size_t)N * K; i++) in[i] = 0.0;
    for (int n = 0; n < N; n++) in[(size_t)n * K + 0] = (double)(n + 1);

    printf("step 1: about to plan\n"); fflush(stdout);
    stride_plan_t *plan = stride_dct2_auto_plan_wis(N, K, &reg, NULL);
    if (!plan) { printf("plan failed\n"); return 1; }

    printf("step 2: plan OK; inner R2C N=%d K=%zu\n",
           plan->override_data ? ((stride_dct2_data_t *)plan->override_data)->r2c_plan->N : -1,
           plan->override_data ? ((stride_dct2_data_t *)plan->override_data)->r2c_plan->K : 0);
    fflush(stdout);

    printf("step 3: calling stride_execute_dct2\n"); fflush(stdout);
    stride_execute_dct2(plan, in, out);
    printf("step 4: execute returned\n"); fflush(stdout);

    /* Reference for K-column 0 of the input (= [1..8]) */
    double ref[8];
    for (int k = 0; k < N; k++) {
        double s = 0;
        for (int n = 0; n < N; n++)
            s += in[(size_t)n * K + 0] * cos(M_PI * k * (2*n + 1) / (2.0 * N));
        ref[k] = 2.0 * s;
    }

    printf("\n  k    got            ref            diff\n");
    for (int k = 0; k < N; k++) {
        double g = out[(size_t)k * K + 0];
        printf("  %d  %12.6f   %12.6f   %.2e%s\n",
               k, g, ref[k], fabs(g - ref[k]),
               fabs(g - ref[k]) > 1e-12 ? "  MISMATCH" : "");
    }

    stride_plan_destroy(plan);
    return 0;
}
