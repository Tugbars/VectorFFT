/* test_c2c_rt_diag.c -- diagnostic: verify stride_auto_plan's fwd+bwd produces N*input. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "env.h"

static double max_abs_diff(const double *a, const double *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) { double d = fabs(a[i]-b[i]); if (d>m) m=d; }
    return m;
}

int main(void) {
    stride_env_init();
    stride_set_num_threads(1);
    stride_registry_t reg; stride_registry_init(&reg);

    int sizes[] = { 8, 16, 32, 64, 128, 256, 512 };
    size_t Ks[] = { 4, 32, 256 };

    for (int si = 0; si < (int)(sizeof(sizes)/sizeof(sizes[0])); si++) {
        int N = sizes[si];
        for (int ki = 0; ki < (int)(sizeof(Ks)/sizeof(Ks[0])); ki++) {
            size_t K = Ks[ki];
            size_t NK = (size_t)N * K;
            double *r0 = (double *)_aligned_malloc(NK*sizeof(double), 64);
            double *i0 = (double *)_aligned_malloc(NK*sizeof(double), 64);
            double *r1 = (double *)_aligned_malloc(NK*sizeof(double), 64);
            double *i1 = (double *)_aligned_malloc(NK*sizeof(double), 64);

            srand(7+N+(int)K);
            for (size_t i = 0; i < NK; i++) {
                r0[i] = (double)rand()/RAND_MAX - 0.5;
                i0[i] = (double)rand()/RAND_MAX - 0.5;
            }
            memcpy(r1, r0, NK*sizeof(double));
            memcpy(i1, i0, NK*sizeof(double));

            stride_plan_t *plan = stride_auto_plan(N, K, &reg);
            if (!plan) { printf("N=%-4d K=%-3zu PLAN_FAIL\n", N, K); continue; }
            stride_execute_fwd(plan, r1, i1);
            stride_execute_bwd(plan, r1, i1);
            double inv_N = 1.0 / (double)N;
            for (size_t i = 0; i < NK; i++) { r1[i] *= inv_N; i1[i] *= inv_N; }
            double err_r = max_abs_diff(r0, r1, NK);
            double err_i = max_abs_diff(i0, i1, NK);
            printf("N=%-4d K=%-3zu  err_re=%.2e  err_im=%.2e  stages=%d  %s\n",
                   N, K, err_r, err_i, plan->num_stages,
                   (err_r < 1e-10 && err_i < 1e-10) ? "PASS" : "FAIL");
            stride_plan_destroy(plan);
            _aligned_free(r0); _aligned_free(i0); _aligned_free(r1); _aligned_free(i1);
        }
    }
    return 0;
}
