/*
 * real_fft.c -- VectorFFT R2C / C2R example and correctness test
 *
 * Demonstrates the real-to-complex FFT API:
 *   stride_r2c_auto_plan(N, K, reg)   Create an R2C plan (N must be even)
 *   stride_execute_r2c(plan, in, re, im)  Forward: N reals -> N/2+1 complex
 *   stride_execute_c2r(plan, re, im, out) Backward: N/2+1 complex -> N reals
 *
 * Or in-place via the standard execute functions:
 *   stride_execute_fwd(plan, re, im)  Forward R2C (re has reals in, complex out)
 *   stride_execute_bwd(plan, re, im)  Backward C2R (complex in, reals out in re)
 *
 * Build:
 *   cd build && cmake --build . --config Release
 *   ./bin/real_fft
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../src/stride-fft/core/env.h"
#include "../src/stride-fft/core/planner.h"
#include "../src/stride-fft/core/compat.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Brute-force DFT of real input for reference */
static void dft_real_reference(const double *x, double *Xre, double *Xim,
                                int N, size_t K) {
    int out_len = N / 2 + 1;
    for (int f = 0; f < out_len; f++) {
        for (size_t k = 0; k < K; k++) {
            double sr = 0, si = 0;
            for (int n = 0; n < N; n++) {
                double angle = -2.0 * M_PI * (double)f * (double)n / (double)N;
                sr += x[n * K + k] * cos(angle);
                si += x[n * K + k] * sin(angle);
            }
            Xre[f * K + k] = sr;
            Xim[f * K + k] = si;
        }
    }
}

int main(void) {
    unsigned int saved = stride_env_init();

    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("=== VectorFFT R2C / C2R Test ===\n\n");

    /* Test multiple sizes */
    int test_sizes[] = {8, 16, 64, 100, 200, 256, 1000, 1024, 4096};
    int ntests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    size_t K = 4;  /* small K for correctness test */

    int all_pass = 1;

    for (int ti = 0; ti < ntests; ti++) {
        int N = test_sizes[ti];
        int out_len = N / 2 + 1;
        size_t total_in  = (size_t)N * K;
        size_t total_out = (size_t)out_len * K;

        /* Allocate */
        double *x      = (double *)STRIDE_ALIGNED_ALLOC(64, total_in * sizeof(double));
        double *out_re  = (double *)STRIDE_ALIGNED_ALLOC(64, total_in * sizeof(double)); /* N*K for in-place */
        double *out_im  = (double *)STRIDE_ALIGNED_ALLOC(64, total_out * sizeof(double));
        double *ref_re  = (double *)STRIDE_ALIGNED_ALLOC(64, total_out * sizeof(double));
        double *ref_im  = (double *)STRIDE_ALIGNED_ALLOC(64, total_out * sizeof(double));
        double *roundtrip = (double *)STRIDE_ALIGNED_ALLOC(64, total_in * sizeof(double));

        /* Fill with test signal: sum of cosines at bins 1, 3, 7 */
        for (int n = 0; n < N; n++) {
            double val = cos(2.0 * M_PI * 1.0 * n / N)
                       + 0.5 * cos(2.0 * M_PI * 3.0 * n / N)
                       + 0.25 * cos(2.0 * M_PI * 7.0 * n / N);
            for (size_t k = 0; k < K; k++)
                x[n * K + k] = val;
        }

        /* Create R2C plan */
        stride_plan_t *plan = stride_r2c_auto_plan(N, K, &reg);
        if (!plan) {
            printf("  N=%-5d PLAN FAILED\n", N);
            all_pass = 0;
            continue;
        }

        /* Forward R2C */
        stride_execute_r2c(plan, x, out_re, out_im);

        /* Reference DFT */
        dft_real_reference(x, ref_re, ref_im, N, K);

        /* Compare forward */
        double max_fwd_err = 0;
        for (int f = 0; f < out_len; f++) {
            for (size_t k = 0; k < K; k++) {
                double er = fabs(out_re[f*K+k] - ref_re[f*K+k]);
                double ei = fabs(out_im[f*K+k] - ref_im[f*K+k]);
                if (er > max_fwd_err) max_fwd_err = er;
                if (ei > max_fwd_err) max_fwd_err = ei;
            }
        }

        /* Backward C2R (roundtrip) */
        stride_execute_c2r(plan, out_re, out_im, roundtrip);
        /* Normalize */
        for (size_t i = 0; i < total_in; i++)
            roundtrip[i] /= N;

        double max_rt_err = 0;
        for (int n = 0; n < N; n++) {
            for (size_t k = 0; k < K; k++) {
                double err = fabs(roundtrip[n*K+k] - x[n*K+k]);
                if (err > max_rt_err) max_rt_err = err;
            }
        }

        int pass = (max_fwd_err < 1e-8) && (max_rt_err < 1e-12);
        printf("  N=%-5d fwd_err=%.2e  rt_err=%.2e  %s\n",
               N, max_fwd_err, max_rt_err, pass ? "OK" : "FAIL");
        if (!pass) all_pass = 0;

        stride_plan_destroy(plan);
        STRIDE_ALIGNED_FREE(x);
        STRIDE_ALIGNED_FREE(out_re);
        STRIDE_ALIGNED_FREE(out_im);
        STRIDE_ALIGNED_FREE(ref_re);
        STRIDE_ALIGNED_FREE(ref_im);
        STRIDE_ALIGNED_FREE(roundtrip);
    }

    /* ── Benchmark: R2C vs complex FFT ── */
    printf("\n=== R2C vs Complex FFT Benchmark ===\n\n");
    stride_pin_thread(0);  /* Pin to P-core 0 for stable timing */
    {
        int bench_sizes[] = {256, 1000, 4096, 10000};
        int nbench = sizeof(bench_sizes) / sizeof(bench_sizes[0]);
        size_t BK = 256;

        printf("%-8s %12s %12s %8s\n", "N", "r2c_ns", "complex_ns", "ratio");
        printf("-------+-------------+-------------+--------\n");

        for (int bi = 0; bi < nbench; bi++) {
            int BN = bench_sizes[bi];
            size_t btotal = (size_t)BN * BK;
            int bout_len = BN / 2 + 1;

            double *bx   = (double *)STRIDE_ALIGNED_ALLOC(64, btotal * sizeof(double));
            double *bre   = (double *)STRIDE_ALIGNED_ALLOC(64, btotal * sizeof(double));
            double *bim   = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)bout_len * BK * sizeof(double));
            double *cre   = (double *)STRIDE_ALIGNED_ALLOC(64, btotal * sizeof(double));
            double *cim   = (double *)STRIDE_ALIGNED_ALLOC(64, btotal * sizeof(double));

            for (size_t i = 0; i < btotal; i++) {
                bx[i] = (double)rand() / RAND_MAX;
                cre[i] = bx[i];
                cim[i] = 0.0;
            }

            stride_plan_t *r2c_plan = stride_r2c_auto_plan(BN, BK, &reg);
            stride_plan_t *c2c_plan = stride_auto_plan(BN, BK, &reg);

            if (!r2c_plan || !c2c_plan) {
                printf("%-8d  PLAN FAILED\n", BN);
                continue;
            }

            /* Warmup */
            for (int w = 0; w < 20; w++) {
                memcpy(bre, bx, btotal * sizeof(double));
                stride_execute_fwd(r2c_plan, bre, bim);
                stride_execute_fwd(c2c_plan, cre, cim);
            }

            /* Bench R2C (in-place, no restore — measures pure execute time) */
            int reps = 2000;
            double t0 = now_ns();
            for (int r = 0; r < reps; r++)
                stride_execute_fwd(r2c_plan, bre, bim);
            double r2c_ns = (now_ns() - t0) / reps;

            /* Bench complex */
            t0 = now_ns();
            for (int r = 0; r < reps; r++)
                stride_execute_fwd(c2c_plan, cre, cim);
            double c2c_ns = (now_ns() - t0) / reps;

            printf("%-8d %12.0f %12.0f %7.2fx\n", BN, r2c_ns, c2c_ns, c2c_ns / r2c_ns);

            stride_plan_destroy(r2c_plan);
            stride_plan_destroy(c2c_plan);
            STRIDE_ALIGNED_FREE(bx);
            STRIDE_ALIGNED_FREE(bre);
            STRIDE_ALIGNED_FREE(bim);
            STRIDE_ALIGNED_FREE(cre);
            STRIDE_ALIGNED_FREE(cim);
        }
    }

    printf("\n%s\n", all_pass ? "All tests PASSED" : "Some tests FAILED");

    stride_env_restore(saved);
    return all_pass ? 0 : 1;
}
