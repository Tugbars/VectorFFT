/**
 * bench_2d.c — 2D FFT benchmark: VectorFFT (tiled, multi-threaded) vs MKL
 *
 * Tests 1, 2, 4, 8 threads. Includes roundtrip correctness check.
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

static double roundtrip_err(const double *orig_re, const double *orig_im,
                            double *re, double *im, size_t total,
                            stride_plan_t *plan) {
    memcpy(re, orig_re, total * sizeof(double));
    memcpy(im, orig_im, total * sizeof(double));
    stride_execute_fwd(plan, re, im);
    stride_execute_bwd(plan, re, im);
    double N = (double)plan->N, mx = 0;
    for (size_t i = 0; i < total; i++) {
        double d = fabs(re[i] / N - orig_re[i]);
        if (d > mx) mx = d;
        d = fabs(im[i] / N - orig_im[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static double bench_fwd(stride_plan_t *plan, double *re, double *im,
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

int main(void) {
    stride_env_init();
    stride_pin_thread(0);

    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("=== 2D FFT Multi-threaded Benchmark ===\n\n");

    int sizes[][2] = {
        {128, 128},
        {256, 256},
        {512, 512},
        {1024, 1024},
    };
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    int thread_counts[] = {1, 2, 4, 8};
    int nthreads = sizeof(thread_counts) / sizeof(thread_counts[0]);
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

        double us_1t = 0;

        for (int ti = 0; ti < nthreads; ti++) {
            int T = thread_counts[ti];
            stride_set_num_threads(T);

            stride_plan_t *plan = stride_plan_2d(N1, N2, &reg);
            if (!plan) { printf("  T=%d: PLAN FAILED\n", T); continue; }

            double err = roundtrip_err(ref_re, ref_im, re, im, total, plan);
            double us = bench_fwd(plan, re, im, ref_re, ref_im, total, reps);

            if (T == 1) us_1t = us;
            double speedup = (us_1t > 0) ? us_1t / us : 0;

            printf("  T=%d:  %8.1f us  speedup=%.2fx  err=%.1e\n",
                   T, us, speedup, err);

            stride_plan_destroy(plan);
        }

        /* MKL single-threaded for reference */
#ifdef VFFT_HAS_MKL
        {
            mkl_set_num_threads(1);
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
            double mkl_us = (now_ns() - t0) / reps / 1000.0;
            DftiFreeDescriptor(&mkl_h);
            printf("  MKL(1T): %6.1f us\n", mkl_us);
        }
#endif
        printf("\n");

        STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
        STRIDE_ALIGNED_FREE(ref_re); STRIDE_ALIGNED_FREE(ref_im);
    }

    stride_set_num_threads(1);
    printf("Done.\n");
    return 0;
}
