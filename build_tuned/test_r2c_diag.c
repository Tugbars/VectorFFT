/* test_r2c_diag.c — diagnose R2C bug.
 *
 * For a single small case (N=8, K=1):
 *   1. Run R2C forward
 *   2. Compute reference DFT directly from the real input
 *   3. Print both side-by-side to identify which bins are wrong
 *
 * Also dumps the inner plan's factor list for the same case.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "r2c.h"
#include "env.h"

#define N 64
#define K 4

static void direct_real_dft(const double *x, int Nlen, double *X_re, double *X_im) {
    for (int k = 0; k <= Nlen/2; k++) {
        double sr = 0, si = 0;
        for (int n = 0; n < Nlen; n++) {
            double angle = -2.0 * M_PI * (double)k * (double)n / (double)Nlen;
            sr += x[n] * cos(angle);
            si += x[n] * sin(angle);
        }
        X_re[k] = sr;
        X_im[k] = si;
    }
}

int main(void) {
    stride_env_init();
    stride_set_num_threads(1);

    stride_registry_t reg;
    stride_registry_init(&reg);

    /* Use a known signal: x[n] = n (simple ramp) */
    double *real_in = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) real_in[i] = (double)i;

    /* Reference DFT */
    double *ref_re = (double *)malloc((N/2 + 1) * sizeof(double));
    double *ref_im = (double *)malloc((N/2 + 1) * sizeof(double));
    direct_real_dft(real_in, N, ref_re, ref_im);

    /* R2C plan + execute */
    stride_plan_t *plan = stride_r2c_auto_plan(N, K, &reg);
    if (!plan) { printf("plan failed\n"); return 1; }

    /* Plan inspection */
    stride_r2c_data_t *d = (stride_r2c_data_t *)plan->override_data;
    printf("N=%d halfN=%d K=%d B=%zu\n", N, d->half_N, K, d->B);
    printf("inner num_stages=%d factors=", d->inner->num_stages);
    for (int s = 0; s < d->inner->num_stages; s++)
        printf("%d%s", d->inner->factors[s],
               s == d->inner->num_stages - 1 ? "" : "x");
    printf(" use_dif_forward=%d\n", d->inner->use_dif_forward);

    printf("perm: ");
    for (int i = 0; i < d->half_N; i++) printf("%d ", d->perm[i]);
    printf("\n");
    printf("iperm: ");
    for (int i = 0; i < d->half_N; i++) printf("%d ", d->iperm[i]);
    printf("\n");

    /* Allocate buffers, real-sized (in-place R2C convention) — full N*K
     * because the convenience wrapper memcpy's N*K reals into out_re. */
    double *real_in_batch = (double *)_aligned_malloc(N * K * sizeof(double), 64);
    double *out_re = (double *)_aligned_malloc(N * K * sizeof(double), 64);
    double *out_im = (double *)_aligned_malloc(N * K * sizeof(double), 64);

    /* Replicate the same real_in across all K columns */
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
            real_in_batch[n * K + k] = real_in[n];

    stride_execute_r2c(plan, real_in_batch, out_re, out_im);

    /* Print bin values for K-column 0 (all columns should match since input is replicated) */
    printf("\n  k    ref_re        ref_im        got_re        got_im\n");
    for (int k = 0; k <= N/2; k++) {
        double got_re = out_re[(size_t)k * K + 0];
        double got_im = out_im[(size_t)k * K + 0];
        printf("  %d  %12.6f  %12.6f  %12.6f  %12.6f%s\n",
               k, ref_re[k], ref_im[k], got_re, got_im,
               (fabs(got_re - ref_re[k]) > 1e-10 ||
                fabs(got_im - ref_im[k]) > 1e-10) ? "  <-- MISMATCH" : "");
    }
    /* Now C2R roundtrip: take freq output, run C2R bwd, normalize, check vs orig. */
    double *real_back = (double *)_aligned_malloc(N * K * sizeof(double), 64);
    stride_execute_c2r(plan, out_re, out_im, real_back);
    double inv_N = 1.0 / (double)N;
    for (size_t i = 0; i < (size_t)N * K; i++) real_back[i] *= inv_N;

    printf("\n  n  orig         back         diff\n");
    int n_bad = 0;
    for (int n = 0; n < N; n++) {
        double orig = real_in[n];
        double back = real_back[(size_t)n * K + 0];
        double diff = back - orig;
        if (fabs(diff) > 1e-10) n_bad++;
        if (n < 10 || fabs(diff) > 1e-10)
            printf("  %2d  %12.6f  %12.6f  %12.6e%s\n",
                   n, orig, back, diff, fabs(diff) > 1e-10 ? "  <-- BAD" : "");
    }
    printf("  (%d bad of %d)\n", n_bad, N);

    _aligned_free(real_back);
    _aligned_free(real_in_batch);
    _aligned_free(out_re); _aligned_free(out_im);
    free(real_in); free(ref_re); free(ref_im);
    stride_plan_destroy(plan);
    return 0;
}
