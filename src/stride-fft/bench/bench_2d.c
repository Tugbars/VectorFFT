/**
 * bench_2d.c — 2D FFT benchmark: VectorFFT (Bailey) vs MKL
 *
 * Uses exhaustive sub-plan search (default for 2D).
 * Includes roundtrip correctness check and timing breakdown.
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

int main(void) {
    stride_env_init();
    stride_pin_thread(0);
    stride_set_num_threads(1);
#ifdef VFFT_HAS_MKL
    mkl_set_num_threads(1);
#endif

    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("=== 2D FFT Benchmark: VectorFFT (Bailey+exhaustive) vs MKL ===\n\n");

    int sizes[][2] = {
        {32, 32},
        {64, 64},
        {128, 128},
        {256, 256},
        {512, 512},
        {1024, 1024},
        {64, 128},
        {128, 256},
        {100, 200},
    };
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    int reps = 200;

    printf("%-12s %10s %10s %10s %10s %8s %10s\n",
           "Size", "vfft_us", "col_us", "tp+row_us", "mkl_us", "ratio", "err");
    printf("------------+----------+----------+----------+----------+--------+----------\n");

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

        /* ── VectorFFT (exhaustive by default) ── */
        stride_plan_t *plan = stride_plan_2d(N1, N2, &reg);
        if (!plan) {
            char label[32];
            snprintf(label, sizeof(label), "%dx%d", N1, N2);
            printf("%-12s  PLAN FAILED\n", label);
            STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
            STRIDE_ALIGNED_FREE(ref_re); STRIDE_ALIGNED_FREE(ref_im);
            continue;
        }

        /* Print factorizations */
        {
            stride_fft2d_data_t *d = (stride_fft2d_data_t *)plan->override_data;
            printf("  %dx%d col: ", N1, N2);
            for (int s = 0; s < d->plan_col->num_stages; s++)
                printf("%d%s", d->plan_col->factors[s], s < d->plan_col->num_stages-1 ? "x" : "");
            printf("  row: ");
            for (int s = 0; s < d->plan_row->num_stages; s++)
                printf("%d%s", d->plan_row->factors[s], s < d->plan_row->num_stages-1 ? "x" : "");
            printf("\n");
        }

        /* Correctness */
        double err = max_roundtrip_err(ref_re, ref_im, re, im, total, plan);

        /* Bench total */
        memcpy(re, ref_re, total * sizeof(double));
        memcpy(im, ref_im, total * sizeof(double));
        for (int w = 0; w < 20; w++)
            stride_execute_fwd(plan, re, im);
        double t0 = now_ns();
        for (int r = 0; r < reps; r++)
            stride_execute_fwd(plan, re, im);
        double vfft_us = (now_ns() - t0) / reps / 1000.0;

        /* Bench column FFT only (for timing split) */
        stride_plan_t *plan_col_only = stride_exhaustive_plan(N1, (size_t)N2, &reg);
        if (!plan_col_only) plan_col_only = stride_auto_plan(N1, (size_t)N2, &reg);
        for (int w = 0; w < 20; w++)
            stride_execute_fwd(plan_col_only, re, im);
        t0 = now_ns();
        for (int r = 0; r < reps; r++)
            stride_execute_fwd(plan_col_only, re, im);
        double col_us = (now_ns() - t0) / reps / 1000.0;
        double tp_row_us = vfft_us - col_us;
        stride_plan_destroy(plan_col_only);

        stride_plan_destroy(plan);

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
        t0 = now_ns();
        for (int r = 0; r < reps; r++)
            DftiComputeForward(mkl_h, re, im);
        mkl_us = (now_ns() - t0) / reps / 1000.0;

        DftiFreeDescriptor(&mkl_h);
#endif

        char label[32];
        snprintf(label, sizeof(label), "%dx%d", N1, N2);
        if (mkl_us > 0)
            printf("%-12s %9.1f %9.1f %9.1f %9.1f %7.2fx %9.1e\n",
                   label, vfft_us, col_us, tp_row_us, mkl_us,
                   mkl_us / vfft_us, err);
        else
            printf("%-12s %9.1f %9.1f %9.1f %9s %7s %9.1e\n",
                   label, vfft_us, col_us, tp_row_us, "N/A", "N/A", err);

        STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
        STRIDE_ALIGNED_FREE(ref_re); STRIDE_ALIGNED_FREE(ref_im);
    }

    printf("\nratio = MKL/ours (>1 = we're faster)\n");
    printf("Done.\n");
    return 0;
}
