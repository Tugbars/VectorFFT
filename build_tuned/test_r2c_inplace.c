/* test_r2c_inplace.c — verify R2C fwd/bwd works via direct stride_execute_*
 * (in-place, no convenience wrapper). N*K-sized buffer for both real input
 * AND freq output AND time output.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "planner.h"
#include "r2c.h"
#include "env.h"

int main(void) {
    stride_env_init();
    stride_set_num_threads(8);   /* Plan with T=8 to match bench_mt_overrides */

    stride_registry_t reg;
    stride_registry_init(&reg);

    int N = 1024;
    size_t K = 256;
    size_t NK = (size_t)N * K;

    double *re = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *im = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *orig = (double *)_aligned_malloc(NK * sizeof(double), 64);

    srand(42);
    for (size_t i = 0; i < NK; i++) re[i] = (double)rand() / RAND_MAX - 0.5;
    memcpy(orig, re, NK * sizeof(double));
    memset(im, 0, NK * sizeof(double));

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);
    stride_wisdom_load(&wis, "vfft_wisdom_tuned.txt");
    stride_plan_t *plan = stride_r2c_wise_plan(N, K, &reg, &wis);
    if (!plan) { printf("plan failed\n"); return 1; }
    stride_r2c_data_t *d = (stride_r2c_data_t *)plan->override_data;
    printf("Plan: N=%d K=%zu B=%zu n_threads=%d inner_factors=", N, K,
           d->B, d->n_threads);
    for (int s = 0; s < d->inner->num_stages; s++) printf("%d ", d->inner->factors[s]);
    printf("inner_dif_fwd=%d\n", d->inner->use_dif_forward);

    /* Build and tear down a complex plan FIRST (to mimic bench_mt_overrides
     * sequence where R2C cells come after several non-R2C cells). */
    {
        for (int i = 0; i < 3; i++) {
            stride_set_num_threads(8);
            stride_plan_t *cplx = stride_auto_plan_wis(1024, 256, &reg, NULL);
            stride_set_num_threads(1);
            double *tre = (double *)_aligned_malloc(NK * sizeof(double), 64);
            double *tim = (double *)_aligned_malloc(NK * sizeof(double), 64);
            memcpy(tre, orig, NK * sizeof(double)); memset(tim, 0, NK * sizeof(double));
            stride_execute_fwd(cplx, tre, tim);
            stride_execute_bwd(cplx, tre, tim);
            _aligned_free(tre); _aligned_free(tim);
            stride_plan_destroy(cplx);
        }
    }

    /* Then: T=8 set BEFORE the R2C plan was already created above.
     * Emulate the bench's order exactly: set T=1, then execute. */
    stride_set_num_threads(1);
    stride_execute_fwd(plan, re, im);
    stride_execute_bwd(plan, re, im);

    double inv_N = 1.0 / (double)N;
    double max_err = 0;
    for (size_t i = 0; i < NK; i++) {
        re[i] *= inv_N;
        double d = fabs(re[i] - orig[i]);
        if (d > max_err) max_err = d;
    }
    printf("Direct in-place N=%d K=%zu  max_err = %.3e  %s\n",
           N, K, max_err, max_err < 1e-10 ? "PASS" : "FAIL");

    /* For comparison: convenience-wrapper roundtrip on same data */
    memcpy(re, orig, NK * sizeof(double));
    memset(im, 0, NK * sizeof(double));
    double *out_re = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *out_im = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *real_back = (double *)_aligned_malloc(NK * sizeof(double), 64);

    stride_execute_r2c(plan, re, out_re, out_im);
    stride_execute_c2r(plan, out_re, out_im, real_back);
    max_err = 0;
    for (size_t i = 0; i < NK; i++) {
        real_back[i] *= inv_N;
        double d = fabs(real_back[i] - orig[i]);
        if (d > max_err) max_err = d;
    }
    printf("Convenience      N=%d K=%zu  max_err = %.3e  %s\n",
           N, K, max_err, max_err < 1e-10 ? "PASS" : "FAIL");

    _aligned_free(re); _aligned_free(im);
    _aligned_free(orig);
    _aligned_free(out_re); _aligned_free(out_im);
    _aligned_free(real_back);
    stride_plan_destroy(plan);
    return 0;
}
