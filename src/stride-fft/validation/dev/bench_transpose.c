/**
 * bench_transpose.c — Transpose benchmark: SIMD blocked vs naive vs MKL
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../core/compat.h"
#include "../core/env.h"
#include "../core/executor.h"
#include "../core/transpose.h"

#ifdef VFFT_HAS_MKL
#include <mkl.h>
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
    stride_env_init();
    stride_pin_thread(0);

    printf("=== Transpose Benchmark ===\n\n");

    int sizes[][2] = {
        {64, 64}, {128, 128}, {256, 256}, {512, 512}, {1024, 1024},
        {64, 128}, {128, 256}, {256, 512}, {100, 200},
    };
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    int reps = 500;

    printf("%-12s %10s %10s %10s %8s\n",
           "Size", "simd_us", "mkl_us", "err", "vs_mkl");
    printf("------------+----------+----------+----------+--------\n");

    for (int si = 0; si < nsizes; si++) {
        int N1 = sizes[si][0], N2 = sizes[si][1];
        size_t total = (size_t)N1 * N2;

        double *src = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        double *dst_ref = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        double *dst_out = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));

        for (size_t i = 0; i < total; i++)
            src[i] = (double)rand() / RAND_MAX;

        /* Correctness */
        stride_transpose(src, N2, dst_ref, N1, N1, N2);
        double err = 0;
        for (int i = 0; i < N1; i++)
            for (int j = 0; j < N2; j++) {
                double d = dst_ref[j * N1 + i] - src[i * N2 + j];
                if (d < 0) d = -d;
                if (d > err) err = d;
            }

        /* Warmup */
        for (int w = 0; w < 20; w++)
            stride_transpose(src, N2, dst_out, N1, N1, N2);

        /* Bench SIMD */
        double t0 = now_ns();
        for (int r = 0; r < reps; r++)
            stride_transpose(src, N2, dst_out, N1, N1, N2);
        double simd_us = (now_ns() - t0) / reps / 1000.0;

        /* Bench MKL */
        double mkl_us = 0;
#ifdef VFFT_HAS_MKL
        for (int w = 0; w < 20; w++)
            mkl_domatcopy('R', 'T', N1, N2, 1.0, src, N2, dst_out, N1);
        t0 = now_ns();
        for (int r = 0; r < reps; r++)
            mkl_domatcopy('R', 'T', N1, N2, 1.0, src, N2, dst_out, N1);
        mkl_us = (now_ns() - t0) / reps / 1000.0;
#endif

        char label[32];
        snprintf(label, sizeof(label), "%dx%d", N1, N2);
        printf("%-12s %9.1f %9.1f %9.1e %7.2fx\n",
               label, simd_us, mkl_us, err,
               mkl_us > 0 ? mkl_us / simd_us : 0);

        STRIDE_ALIGNED_FREE(src);
        STRIDE_ALIGNED_FREE(dst_ref);
        STRIDE_ALIGNED_FREE(dst_out);
    }

    printf("\nvs_mkl = MKL/ours (>1 = we're faster)\n");
    printf("Done.\n");
    return 0;
}
