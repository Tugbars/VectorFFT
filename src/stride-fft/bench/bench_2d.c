/**
 * bench_2d.c — 2D FFT benchmark: VectorFFT vs MKL
 *
 * Tests square and rectangular 2D FFTs.
 * Split-complex layout for VectorFFT, split-complex (DFTI_REAL_REAL) for MKL.
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

int main(void) {
    stride_env_init();
    stride_pin_thread(0);
    stride_set_num_threads(1);
#ifdef VFFT_HAS_MKL
    mkl_set_num_threads(1);
#endif

    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("=== 2D FFT Benchmark: VectorFFT vs MKL (both single-threaded) ===\n\n");

    int sizes[][2] = {
        {32, 32},
        {64, 64},
        {128, 128},
        {256, 256},
        {512, 512},
        {64, 128},
        {128, 256},
        {100, 200},
    };
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    int reps = 200;

    printf("%-12s %10s %10s %10s %10s %8s\n", "Size", "vfft_us", "ax0_us", "ax1_us", "mkl_us", "ratio");
    printf("------------+----------+----------+----------+----------+--------\n");

    for (int si = 0; si < nsizes; si++) {
        int N1 = sizes[si][0], N2 = sizes[si][1];
        size_t total = (size_t)N1 * N2;

        double *re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        double *im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));

        /* Fill with random data */
        for (size_t i = 0; i < total; i++) {
            re[i] = (double)rand() / RAND_MAX;
            im[i] = (double)rand() / RAND_MAX;
        }

        /* ── VectorFFT ── */
        stride_plan_t *plan = stride_plan_2d(N1, N2, &reg);
        if (!plan) {
            printf("%-12s  PLAN FAILED\n", "");
            STRIDE_ALIGNED_FREE(re);
            STRIDE_ALIGNED_FREE(im);
            continue;
        }

        /* Also create a 1D plan for axis-0 only, to measure axis-0 vs axis-1 */
        stride_plan_t *plan_ax0 = stride_auto_plan(N1, (size_t)N2, &reg);

        /* Warmup */
        for (int w = 0; w < 20; w++)
            stride_execute_fwd(plan, re, im);

        /* Total 2D time */
        double t0 = now_ns();
        for (int r = 0; r < reps; r++)
            stride_execute_fwd(plan, re, im);
        double vfft_us = (now_ns() - t0) / reps / 1000.0;

        /* Axis-0 only time */
        t0 = now_ns();
        for (int r = 0; r < reps; r++)
            stride_execute_fwd(plan_ax0, re, im);
        double ax0_us = (now_ns() - t0) / reps / 1000.0;

        double ax1_us = vfft_us - ax0_us;

        stride_plan_destroy(plan_ax0);
        stride_plan_destroy(plan);

#ifdef VFFT_HAS_MKL
        /* ── MKL ── */
        /* Re-fill data */
        for (size_t i = 0; i < total; i++) {
            re[i] = (double)rand() / RAND_MAX;
            im[i] = (double)rand() / RAND_MAX;
        }

        DFTI_DESCRIPTOR_HANDLE mkl_h = NULL;
        MKL_LONG dims[2] = {N1, N2};
        MKL_LONG strides[3] = {0, N2, 1};

        DftiCreateDescriptor(&mkl_h, DFTI_DOUBLE, DFTI_COMPLEX, 2, dims);
        DftiSetValue(mkl_h, DFTI_PLACEMENT, DFTI_INPLACE);
        DftiSetValue(mkl_h, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
        DftiSetValue(mkl_h, DFTI_INPUT_STRIDES, strides);
        DftiSetValue(mkl_h, DFTI_OUTPUT_STRIDES, strides);
        DftiCommitDescriptor(mkl_h);

        /* Warmup */
        for (int w = 0; w < 20; w++)
            DftiComputeForward(mkl_h, re, im);

        t0 = now_ns();
        for (int r = 0; r < reps; r++)
            DftiComputeForward(mkl_h, re, im);
        double mkl_us = (now_ns() - t0) / reps / 1000.0;

        DftiFreeDescriptor(&mkl_h);

        char label[32];
        snprintf(label, sizeof(label), "%dx%d", N1, N2);
        printf("%-12s %9.1f %9.1f %9.1f %9.1f %7.2fx\n", label, vfft_us, ax0_us, ax1_us, mkl_us, mkl_us / vfft_us);
#else
        {
            char label[32];
            snprintf(label, sizeof(label), "%dx%d", N1, N2);
            printf("%-12s %9.1f %9.1f %9.1f %9s %7s\n", label, vfft_us, ax0_us, ax1_us, "N/A", "N/A");
        }
#endif

        STRIDE_ALIGNED_FREE(re);
        STRIDE_ALIGNED_FREE(im);
    }

    printf("\nDone.\n");
    return 0;
}
