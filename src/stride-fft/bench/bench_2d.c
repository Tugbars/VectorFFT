/**
 * bench_2d.c — 2D FFT benchmark: VectorFFT (Bailey) vs MKL
 *
 * Tests heuristic vs exhaustive planner, compares against MKL.
 * Includes roundtrip correctness check.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../core/compat.h"
#include "../core/env.h"
#include "../core/planner.h"

#ifdef VFFT_HAS_MKL
#include <mkl_dfti.h>
#include <mkl_service.h>
#endif

static double max_roundtrip_err(const double *orig_re, const double *orig_im,
                                double *re, double *im, size_t total,
                                stride_plan_t *plan) {
    memcpy(re, orig_re, total * sizeof(double));
    memcpy(im, orig_im, total * sizeof(double));
    stride_execute_fwd(plan, re, im);
    stride_execute_bwd(plan, re, im);
    double N = (double)plan->N;
    double mx = 0;
    for (size_t i = 0; i < total; i++) {
        double dr = fabs(re[i] / N - orig_re[i]);
        double di = fabs(im[i] / N - orig_im[i]);
        if (dr > mx) mx = dr;
        if (di > mx) mx = di;
    }
    return mx;
}

static double bench_plan(stride_plan_t *plan, double *re, double *im,
                         const double *ref_re, const double *ref_im,
                         size_t total, int reps) {
    memcpy(re, ref_re, total * sizeof(double));
    memcpy(im, ref_im, total * sizeof(double));
    for (int w = 0; w < 20; w++)
        stride_execute_fwd(plan, re, im);
    double t0 = now_ns();
    for (int r = 0; r < reps; r++)
        stride_execute_fwd(plan, re, im);
    return (now_ns() - t0) / reps / 1000.0;
}

static void print_plan_factors(stride_plan_t *plan, const char *label) {
    if (plan->override_fwd) {
        /* 2D plan — print sub-plan factors */
        stride_fft2d_data_t *d = (stride_fft2d_data_t *)plan->override_data;
        printf("  %s col [N=%d,K=%zu]: ", label, d->plan_col->N, d->plan_col->K);
        for (int s = 0; s < d->plan_col->num_stages; s++)
            printf("%d%s", d->plan_col->factors[s], s < d->plan_col->num_stages-1 ? "×" : "");
        printf("\n");
        printf("  %s row [N=%d,K=%zu]: ", label, d->plan_row->N, d->plan_row->K);
        for (int s = 0; s < d->plan_row->num_stages; s++)
            printf("%d%s", d->plan_row->factors[s], s < d->plan_row->num_stages-1 ? "×" : "");
        printf("\n");
    }
}

int main(void) {
    stride_env_init();
    stride_pin_thread(0);
    stride_set_num_threads(1);
#ifdef VFFT_HAS_MKL
    mkl_set_num_threads(1);
#endif

    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("=== 2D FFT Benchmark: VectorFFT (Bailey) vs MKL ===\n\n");

    int sizes[][2] = {
        {64, 64},
        {128, 128},
        {256, 256},
        {512, 512},
        {1024, 1024},
        {100, 200},
    };
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    int reps = 200;

    for (int si = 0; si < nsizes; si++) {
        int N1 = sizes[si][0], N2 = sizes[si][1];
        size_t total = (size_t)N1 * N2;

        double *re     = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        double *im     = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        double *ref_re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        double *ref_im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));

        for (size_t i = 0; i < total; i++) {
            ref_re[i] = (double)rand() / RAND_MAX;
            ref_im[i] = (double)rand() / RAND_MAX;
        }

        char label[32];
        snprintf(label, sizeof(label), "%dx%d", N1, N2);
        printf("── %s ──\n", label);

        /* ── Heuristic plan ── */
        stride_plan_t *plan_h = stride_plan_2d(N1, N2, &reg);
        double err_h = max_roundtrip_err(ref_re, ref_im, re, im, total, plan_h);
        double us_h = bench_plan(plan_h, re, im, ref_re, ref_im, total, reps);
        print_plan_factors(plan_h, "heur");

        /* ── Exhaustive plan ── */
        printf("  (running exhaustive search...)\n");
        stride_plan_t *plan_e = stride_plan_2d_measure(N1, N2, &reg);
        double err_e = max_roundtrip_err(ref_re, ref_im, re, im, total, plan_e);
        double us_e = bench_plan(plan_e, re, im, ref_re, ref_im, total, reps);
        print_plan_factors(plan_e, "exh ");

        /* ── MKL ── */
        double mkl_us = 0;
#ifdef VFFT_HAS_MKL
        memcpy(re, ref_re, total * sizeof(double));
        memcpy(im, ref_im, total * sizeof(double));

        DFTI_DESCRIPTOR_HANDLE mkl_h = NULL;
        MKL_LONG dims[2] = {N1, N2};
        MKL_LONG strides[3] = {0, N2, 1};

        DftiCreateDescriptor(&mkl_h, DFTI_DOUBLE, DFTI_COMPLEX, 2, dims);
        DftiSetValue(mkl_h, DFTI_PLACEMENT, DFTI_INPLACE);
        DftiSetValue(mkl_h, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
        DftiSetValue(mkl_h, DFTI_INPUT_STRIDES, strides);
        DftiSetValue(mkl_h, DFTI_OUTPUT_STRIDES, strides);
        DftiCommitDescriptor(mkl_h);

        for (int w = 0; w < 20; w++)
            DftiComputeForward(mkl_h, re, im);
        double t0 = now_ns();
        for (int r = 0; r < reps; r++)
            DftiComputeForward(mkl_h, re, im);
        mkl_us = (now_ns() - t0) / reps / 1000.0;

        DftiFreeDescriptor(&mkl_h);
#endif

        printf("  heuristic:  %8.1f us  err=%.1e\n", us_h, err_h);
        printf("  exhaustive: %8.1f us  err=%.1e", us_e, err_e);
        if (us_h > 0) printf("  (%.0f%% of heur)", 100.0 * us_e / us_h);
        printf("\n");
        if (mkl_us > 0) {
            printf("  MKL:        %8.1f us\n", mkl_us);
            printf("  vs MKL:     heur=%.2fx  exh=%.2fx\n",
                   mkl_us / us_h, mkl_us / us_e);
        }
        printf("\n");

        stride_plan_destroy(plan_h);
        stride_plan_destroy(plan_e);
        STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
        STRIDE_ALIGNED_FREE(ref_re); STRIDE_ALIGNED_FREE(ref_im);
    }

    printf("Done.\n");
    return 0;
}
