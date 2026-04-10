/**
 * api_test.c — Test of the vfft public C API.
 * Links against libvfft only — no internal headers.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    /* ── C2C roundtrip ── */
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
        vfft_execute_fwd(p, re, im);
        vfft_execute_bwd_normalized(p, re, im);

        double err = max_err(re, ref, NK);
        printf("  C2C roundtrip N=%d K=%zu  err=%.2e  %s\n", N, K, err, err < 1e-10 ? "OK" : "FAIL");
        if (err >= 1e-10) all_pass = 0;

        vfft_destroy(p);
        vfft_free(re); vfft_free(im); vfft_free(ref);
    }

    /* ── R2C roundtrip (same pattern as real_fft.c) ── */
    {
        int N = 256;
        size_t K = 4;
        size_t NK = (size_t)N * K;
        size_t halfN1K = (size_t)(N / 2 + 1) * K;

        double *x       = (double *)vfft_alloc(NK * sizeof(double));
        double *out_re   = (double *)vfft_alloc(NK * sizeof(double));  /* N*K for workspace */
        double *out_im   = (double *)vfft_alloc(halfN1K * sizeof(double));
        double *roundtrip = (double *)vfft_alloc(NK * sizeof(double));

        /* Same test signal as real_fft.c */
        for (int n = 0; n < N; n++) {
            double val = cos(2.0 * M_PI * 1.0 * n / N)
                       + 0.5 * cos(2.0 * M_PI * 3.0 * n / N)
                       + 0.25 * cos(2.0 * M_PI * 7.0 * n / N);
            for (size_t k = 0; k < K; k++)
                x[n * K + k] = val;
        }

        vfft_plan p = vfft_plan_r2c(N, K);
        if (!p) { printf("  R2C plan FAILED\n"); return 1; }

        vfft_execute_r2c(p, x, out_re, out_im);
        vfft_execute_c2r(p, out_re, out_im, roundtrip);

        for (size_t i = 0; i < NK; i++) roundtrip[i] /= N;

        double err = max_err(x, roundtrip, NK);
        printf("  R2C roundtrip N=%d K=%zu  err=%.2e  %s\n", N, K, err, err < 1e-10 ? "OK" : "FAIL");
        if (err >= 1e-10) all_pass = 0;

        vfft_destroy(p);
        vfft_free(x); vfft_free(out_re); vfft_free(out_im); vfft_free(roundtrip);
    }

    /* ── Deinterleave roundtrip ── */
    {
        size_t count = 1024;
        double *il = (double *)vfft_alloc(2 * count * sizeof(double));
        double *re = (double *)vfft_alloc(count * sizeof(double));
        double *im = (double *)vfft_alloc(count * sizeof(double));
        double *out = (double *)vfft_alloc(2 * count * sizeof(double));

        for (size_t i = 0; i < 2 * count; i++) il[i] = (double)rand() / RAND_MAX;
        vfft_deinterleave(il, re, im, count);
        vfft_reinterleave(re, im, out, count);

        double err = max_err(il, out, 2 * count);
        printf("  Deinterleave roundtrip  err=%.2e  %s\n", err, err == 0.0 ? "OK" : "FAIL");
        if (err != 0.0) all_pass = 0;

        vfft_free(il); vfft_free(re); vfft_free(im); vfft_free(out);
    }

    /* ── 2D FFT roundtrip ── */
    {
        int sizes[][2] = {{8, 8}, {16, 32}, {64, 64}, {128, 256}, {100, 200}};
        int nsizes = sizeof(sizes) / sizeof(sizes[0]);

        for (int si = 0; si < nsizes; si++) {
            int N1 = sizes[si][0], N2 = sizes[si][1];
            size_t total = (size_t)N1 * N2;

            double *re  = (double *)vfft_alloc(total * sizeof(double));
            double *im  = (double *)vfft_alloc(total * sizeof(double));
            double *ref = (double *)vfft_alloc(total * sizeof(double));

            for (size_t i = 0; i < total; i++) {
                re[i] = (double)rand() / RAND_MAX;
                im[i] = (double)rand() / RAND_MAX;
                ref[i] = re[i];
            }

            vfft_plan p = vfft_plan_2d(N1, N2);
            if (!p) { printf("  2D %dx%d  PLAN FAILED\n", N1, N2); all_pass = 0; continue; }

            vfft_execute_fwd(p, re, im);
            vfft_execute_bwd_normalized(p, re, im);

            double err = max_err(re, ref, total);
            int ok = err < 1e-8;
            printf("  2D %3dx%-3d roundtrip  err=%.2e  %s\n", N1, N2, err, ok ? "OK" : "FAIL");
            if (!ok) all_pass = 0;

            vfft_destroy(p);
            vfft_free(re); vfft_free(im); vfft_free(ref);
        }
    }

    printf("\n%s\n", all_pass ? "All API tests PASSED" : "Some tests FAILED");
    return all_pass ? 0 : 1;
}
