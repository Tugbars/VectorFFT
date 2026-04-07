/**
 * api_test.c — Minimal test of the vfft public C API.
 *
 * Tests: plan creation, forward, backward normalized, roundtrip, R2C, destroy.
 * Links against libvfft only — no internal headers.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vfft.h"

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

    /* ── C2C roundtrip test ── */
    {
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

    /* ── R2C roundtrip test ── */
    {
        int N = 256;
        size_t K = 64;
        size_t NK = (size_t)N * K;
        size_t halfN1K = (size_t)(N / 2 + 1) * K;

        double *real_in  = (double *)vfft_alloc(NK * sizeof(double));
        double *real_out = (double *)vfft_alloc(NK * sizeof(double));
        double *cre      = (double *)vfft_alloc(halfN1K * sizeof(double));
        double *cim      = (double *)vfft_alloc(halfN1K * sizeof(double));

        for (size_t i = 0; i < NK; i++)
            real_in[i] = (double)rand() / RAND_MAX;

        vfft_plan p = vfft_plan_r2c(N, K);
        if (!p) { printf("R2C plan FAILED\n"); return 1; }

        vfft_execute_r2c(p, real_in, cre, cim);
        vfft_execute_c2r(p, cre, cim, real_out);

        /* C2R gives N * input, so normalize */
        double inv_N = 1.0 / (double)N;
        for (size_t i = 0; i < NK; i++) real_out[i] *= inv_N;

        double err = max_err(real_in, real_out, NK);
        int ok = err < 1e-10;
        printf("  R2C roundtrip N=%d K=%zu  err=%.2e  %s\n", N, K, err, ok ? "OK" : "FAIL");
        if (!ok) all_pass = 0;

        vfft_destroy(p);
        vfft_free(real_in);
        vfft_free(real_out);
        vfft_free(cre);
        vfft_free(cim);
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
