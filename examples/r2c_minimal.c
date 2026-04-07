/* Minimal R2C roundtrip — identical logic to real_fft.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "env.h"
#include "planner.h"

int main(void) {
    stride_env_init();
    stride_registry_t reg;
    stride_registry_init(&reg);

    int N = 256;
    size_t K = 4;
    int out_len = N / 2 + 1;
    size_t total_in  = (size_t)N * K;
    size_t total_out = (size_t)out_len * K;

    double *x     = (double *)STRIDE_ALIGNED_ALLOC(64, total_in * sizeof(double));
    double *ore   = (double *)STRIDE_ALIGNED_ALLOC(64, total_in * sizeof(double));
    double *oim   = (double *)STRIDE_ALIGNED_ALLOC(64, total_out * sizeof(double));
    double *rt    = (double *)STRIDE_ALIGNED_ALLOC(64, total_in * sizeof(double));

    srand(42);
    for (size_t i = 0; i < total_in; i++)
        x[i] = (double)rand() / RAND_MAX;

    stride_plan_t *plan = stride_r2c_auto_plan(N, K, &reg);
    if (!plan) { printf("PLAN FAILED\n"); return 1; }

    stride_execute_r2c(plan, x, ore, oim);
    stride_execute_c2r(plan, ore, oim, rt);
    for (size_t i = 0; i < total_in; i++) rt[i] /= N;

    double mx = 0;
    for (size_t i = 0; i < total_in; i++) {
        double e = fabs(rt[i] - x[i]);
        if (e > mx) mx = e;
    }
    printf("N=%d K=%zu err=%.2e %s\n", N, K, mx, mx < 1e-10 ? "OK" : "FAIL");
    return mx < 1e-10 ? 0 : 1;
}
