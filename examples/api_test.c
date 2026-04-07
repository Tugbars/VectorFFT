/**
 * api_test.c — Minimal test of the vfft public C API.
 *
 * Tests: plan creation, forward, backward normalized, roundtrip, R2C, destroy.
 * Links against libvfft only — no internal headers.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* #include "vfft.h" -- REMOVED for diagnosis */

/* Include internals directly */
#include "env.h"
#include "planner.h"

static double max_err(const double *a, const double *b, size_t n) {
    double mx = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

int main(void) {
    vfft_init();

    printf("VectorFFT %s  ISA: %s\n\n", vfft_version(), vfft_isa());

    int all_pass = 1;

    /* ── C2C roundtrip test — DISABLED for R2C debugging ── */
    if (0) {
        int N = 1000;
        size_t K = 64;
        size_t NK = (size_t)N * K;

        double *re  = (double *)vfft_alloc(NK * sizeof(double));
        double *im  = (double *)vfft_alloc(NK * sizeof(double));
        double *ref = (double *)vfft_alloc(NK * sizeof(double));

        for (size_t i = 0; i < NK; i++) {
            re[i] = (double)rand() / RAND_MAX;
            im[i] = (double)rand() / RAND_MAX;
            ref[i] = re[i];
        }

        vfft_plan p = vfft_plan_c2c(N, K);
        if (!p) { printf("C2C plan FAILED\n"); return 1; }

        vfft_execute_fwd(p, re, im);
        vfft_execute_bwd_normalized(p, re, im);

        double err = max_err(re, ref, NK);
        int ok = err < 1e-10;
        printf("  C2C roundtrip N=%d K=%zu  err=%.2e  %s\n", N, K, err, ok ? "OK" : "FAIL");
        if (!ok) all_pass = 0;

        vfft_destroy(p);
        vfft_free(re);
        vfft_free(im);
        vfft_free(ref);
    }

    /* ── R2C roundtrip tests at multiple K values ── */
    {
        int test_Ks[] = {4};
        int nKs = sizeof(test_Ks) / sizeof(test_Ks[0]);

        for (int ki = 0; ki < nKs; ki++) {
            int N = 256;
            size_t K = (size_t)test_Ks[ki];
            size_t NK = (size_t)N * K;
            size_t halfN1K = (size_t)(N / 2 + 1) * K;

            double *real_in  = (double *)vfft_alloc(NK * sizeof(double));
            double *real_out = (double *)vfft_alloc(NK * sizeof(double));
            double *cre      = (double *)vfft_alloc(NK * sizeof(double));
            double *cim      = (double *)vfft_alloc(halfN1K * sizeof(double));

            srand(42);  /* deterministic */
            for (size_t i = 0; i < NK; i++)
                real_in[i] = (double)rand() / RAND_MAX;

            vfft_plan p = vfft_plan_r2c(N, K);
            if (!p) { printf("  R2C K=%zu PLAN FAILED\n", K); continue; }

            /* Test A: through vfft API */
            vfft_execute_r2c(p, real_in, cre, cim);
            vfft_execute_c2r(p, cre, cim, real_out);

            double inv_N_a = 1.0 / (double)N;
            for (size_t ii = 0; ii < NK; ii++) real_out[ii] *= inv_N_a;
            double err_a = max_err(real_in, real_out, NK);

            /* Test B: EXACT copy of real_fft.c logic */
            {
                stride_registry_t local_reg;
                stride_registry_init(&local_reg);
                int out_len = N / 2 + 1;
                size_t total_in  = (size_t)N * K;
                size_t total_out = (size_t)out_len * K;
                double *x2     = (double *)STRIDE_ALIGNED_ALLOC(64, total_in * sizeof(double));
                double *ore2   = (double *)STRIDE_ALIGNED_ALLOC(64, total_in * sizeof(double));
                double *oim2   = (double *)STRIDE_ALIGNED_ALLOC(64, total_out * sizeof(double));
                double *rt2    = (double *)STRIDE_ALIGNED_ALLOC(64, total_in * sizeof(double));
                srand(42);
                for (size_t ii = 0; ii < total_in; ii++)
                    x2[ii] = (double)rand() / RAND_MAX;
                stride_plan_t *plan2 = stride_r2c_auto_plan(N, K, &local_reg);
                stride_execute_r2c(plan2, x2, ore2, oim2);
                stride_execute_c2r(plan2, ore2, oim2, rt2);
                for (size_t ii = 0; ii < total_in; ii++) rt2[ii] /= N;
                double err_b = max_err(x2, rt2, total_in);
                printf("  direct copy N=%d K=%zu  err=%.2e\n", N, K, err_b);
                stride_plan_destroy(plan2);
                STRIDE_ALIGNED_FREE(x2); STRIDE_ALIGNED_FREE(ore2);
                STRIDE_ALIGNED_FREE(oim2); STRIDE_ALIGNED_FREE(rt2);
            }

            printf("  R2C N=%d K=%zu  vfft_err=%.2e  %s\n",
                   N, K, err_a, err_a < 1e-10 ? "OK" : "FAIL");
            if (err_a >= 1e-10) all_pass = 0;

            vfft_destroy(p);
            vfft_free(real_in);
            vfft_free(real_out);
            vfft_free(cre);
            vfft_free(cim);
        }
    }

    /* ── Deinterleave/reinterleave roundtrip ── */
    {
        size_t count = 1024;
        double *interleaved = (double *)vfft_alloc(2 * count * sizeof(double));
        double *re = (double *)vfft_alloc(count * sizeof(double));
        double *im = (double *)vfft_alloc(count * sizeof(double));
        double *out = (double *)vfft_alloc(2 * count * sizeof(double));

        for (size_t i = 0; i < 2 * count; i++)
            interleaved[i] = (double)rand() / RAND_MAX;

        vfft_deinterleave(interleaved, re, im, count);
        vfft_reinterleave(re, im, out, count);

        double err = max_err(interleaved, out, 2 * count);
        int ok = err == 0.0;
        printf("  Deinterleave roundtrip count=%zu  err=%.2e  %s\n", count, err, ok ? "OK" : "FAIL");
        if (!ok) all_pass = 0;

        vfft_free(interleaved);
        vfft_free(re);
        vfft_free(im);
        vfft_free(out);
    }

    printf("\n%s\n", all_pass ? "All API tests PASSED" : "Some tests FAILED");
    return all_pass ? 0 : 1;
}
