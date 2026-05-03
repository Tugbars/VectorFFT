/* Diag: minimal 2D R2C test with prints. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "fft2d_r2c.h"
#include "env.h"

int main(void) {
    stride_env_init();
    stride_set_num_threads(1);
    stride_registry_t reg; stride_registry_init(&reg);
    stride_wisdom_t wis; stride_wisdom_init(&wis);

    int N1 = 16, N2 = 16;
    size_t real_sz = (size_t)N1 * (size_t)N2;
    size_t cplx_sz = (size_t)N1 * (size_t)(N2 / 2 + 1);

    fprintf(stderr, "[diag] alloc\n"); fflush(stderr);
    double *orig    = (double *)_aligned_malloc(real_sz * sizeof(double), 64);
    double *out_re  = (double *)_aligned_malloc(cplx_sz * sizeof(double), 64);
    double *out_im  = (double *)_aligned_malloc(cplx_sz * sizeof(double), 64);
    double *back    = (double *)_aligned_malloc(real_sz * sizeof(double), 64);
    /* Impulse at (0,0): expect output bins all = 1+0i */
    for (size_t i = 0; i < real_sz; i++) orig[i] = 0.0;
    orig[0] = 1.0;

    fprintf(stderr, "[diag] plan\n"); fflush(stderr);
    stride_plan_t *plan = stride_plan_2d_r2c(N1, N2, &reg);
    if (!plan) { fprintf(stderr, "PLAN_FAIL\n"); return 1; }

    /* Inspect the col FFT plan's factorization. */
    stride_fft2d_r2c_data_t *d = (stride_fft2d_r2c_data_t *)plan->override_data;
    fprintf(stderr, "[diag] col plan: N=%d K=%zu stages=%d factors=",
            d->plan_col->N, d->plan_col->K, d->plan_col->num_stages);
    for (int s = 0; s < d->plan_col->num_stages; s++)
        fprintf(stderr, "%d,", d->plan_col->factors[s]);
    fprintf(stderr, "\n[diag] r2c plan: N=%d K=%zu B=%zu\n",
            ((stride_r2c_data_t *)d->plan_r2c->override_data)->N,
            ((stride_r2c_data_t *)d->plan_r2c->override_data)->K,
            ((stride_r2c_data_t *)d->plan_r2c->override_data)->B);
    fflush(stderr);

    fprintf(stderr, "[diag] orig:");
    for (size_t i = 0; i < real_sz; i++) fprintf(stderr, " %.2f", orig[i]);
    fprintf(stderr, "\n"); fflush(stderr);

    fprintf(stderr, "[diag] fwd\n"); fflush(stderr);
    stride_execute_2d_r2c(plan, orig, out_re, out_im);
    fprintf(stderr, "[diag] fwd done\n");
    fprintf(stderr, "[diag] out_re:");
    for (size_t i = 0; i < cplx_sz; i++) fprintf(stderr, " %.2f", out_re[i]);
    fprintf(stderr, "\n[diag] out_im:");
    for (size_t i = 0; i < cplx_sz; i++) fprintf(stderr, " %.2f", out_im[i]);
    fprintf(stderr, "\n"); fflush(stderr);

    fprintf(stderr, "[diag] bwd\n"); fflush(stderr);
    stride_execute_2d_c2r(plan, out_re, out_im, back);
    fprintf(stderr, "[diag] bwd done\n"); fflush(stderr);

    double inv_N = 1.0 / (double)real_sz;
    double max_err = 0.0;
    for (size_t i = 0; i < real_sz; i++) {
        back[i] *= inv_N;
        double e = fabs(orig[i] - back[i]);
        if (e > max_err) max_err = e;
    }
    fprintf(stderr, "[diag] max_err = %.2e\n", max_err);

    stride_plan_destroy(plan);
    _aligned_free(orig); _aligned_free(out_re); _aligned_free(out_im); _aligned_free(back);
    return 0;
}
