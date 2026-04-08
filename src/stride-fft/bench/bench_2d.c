/**
 * bench_2d.c — 2D FFT benchmark: tiled vs Bailey vs MKL
 *
 * Compares both regimes at every size to find the crossover.
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

/* Build a forced-Bailey plan (always full transpose, even for small sizes) */
static stride_plan_t *_build_bailey(int N1, int N2,
                                     const stride_registry_t *reg) {
    stride_fft2d_data_t *d = (stride_fft2d_data_t *)calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->N1 = N1; d->N2 = N2; d->use_bailey = 1;

    d->plan_col = stride_exhaustive_plan(N1, (size_t)N2, reg);
    if (!d->plan_col) d->plan_col = stride_auto_plan(N1, (size_t)N2, reg);
    if (!d->plan_col) { free(d); return NULL; }

    d->plan_row = stride_exhaustive_plan(N2, (size_t)N1, reg);
    if (!d->plan_row) d->plan_row = stride_auto_plan(N2, (size_t)N1, reg);
    if (!d->plan_row) { stride_plan_destroy(d->plan_col); free(d); return NULL; }

    d->B = (size_t)N1;
    size_t total = (size_t)N1 * N2;
    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));

    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    plan->N = N1 * N2; plan->K = 1; plan->num_stages = 0;
    plan->override_fwd = _fft2d_execute_fwd;
    plan->override_bwd = _fft2d_execute_bwd;
    plan->override_destroy = _fft2d_destroy;
    plan->override_data = d;
    return plan;
}

/* Build a forced-tiled plan with specific B */
static stride_plan_t *_build_tiled(int N1, int N2, size_t B,
                                    const stride_registry_t *reg) {
    stride_fft2d_data_t *d = (stride_fft2d_data_t *)calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->N1 = N1; d->N2 = N2; d->use_bailey = 0; d->B = B;

    d->plan_col = stride_exhaustive_plan(N1, (size_t)N2, reg);
    if (!d->plan_col) d->plan_col = stride_auto_plan(N1, (size_t)N2, reg);
    if (!d->plan_col) { free(d); return NULL; }

    d->plan_row = stride_exhaustive_plan(N2, B, reg);
    if (!d->plan_row) d->plan_row = stride_auto_plan(N2, B, reg);
    if (!d->plan_row) { stride_plan_destroy(d->plan_col); free(d); return NULL; }

    size_t tile_sz = (size_t)N2 * B;
    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64, tile_sz * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64, tile_sz * sizeof(double));

    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    plan->N = N1 * N2; plan->K = 1; plan->num_stages = 0;
    plan->override_fwd = _fft2d_execute_fwd;
    plan->override_bwd = _fft2d_execute_bwd;
    plan->override_destroy = _fft2d_destroy;
    plan->override_data = d;
    return plan;
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

    printf("=== 2D FFT: Tiled vs Bailey vs MKL ===\n\n");

    int sizes[][2] = {
        {32, 32}, {64, 64}, {128, 128}, {256, 256},
        {512, 512}, {1024, 1024},
        {64, 128}, {128, 256}, {100, 200},
    };
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    size_t tile_Bs[] = {8, 16, 32, 64};
    int ntiles = sizeof(tile_Bs) / sizeof(tile_Bs[0]);
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

        /* Bailey */
        stride_plan_t *pb = _build_bailey(N1, N2, &reg);
        double err_b = roundtrip_err(ref_re, ref_im, re, im, total, pb);
        double us_b = bench_fwd(pb, re, im, ref_re, ref_im, total, reps);
        stride_plan_destroy(pb);
        printf("  Bailey (K=%d):     %8.1f us  err=%.1e\n", N1, us_b, err_b);

        /* Tiled at various B */
        double best_tiled = 1e18;
        size_t best_B = 0;
        for (int ti = 0; ti < ntiles; ti++) {
            size_t B = tile_Bs[ti];
            if (B > (size_t)N1) continue;

            stride_plan_t *pt = _build_tiled(N1, N2, B, &reg);
            if (!pt) continue;
            double err_t = roundtrip_err(ref_re, ref_im, re, im, total, pt);
            double us_t = bench_fwd(pt, re, im, ref_re, ref_im, total, reps);
            stride_plan_destroy(pt);
            printf("  Tiled  (B=%3zu):   %8.1f us  err=%.1e\n", B, us_t, err_t);

            if (us_t < best_tiled) { best_tiled = us_t; best_B = B; }
        }

        /* MKL */
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

        /* Summary */
        double best_us = (best_tiled < us_b) ? best_tiled : us_b;
        const char *winner = (best_tiled < us_b) ? "tiled" : "bailey";
        size_t winner_K = (best_tiled < us_b) ? best_B : (size_t)N1;
        printf("  >> Best: %s (K=%zu) = %.1f us", winner, winner_K, best_us);
        if (mkl_us > 0)
            printf("  |  MKL: %.1f us  |  ratio: %.2fx",
                   mkl_us, mkl_us / best_us);
        printf("\n\n");

        STRIDE_ALIGNED_FREE(re); STRIDE_ALIGNED_FREE(im);
        STRIDE_ALIGNED_FREE(ref_re); STRIDE_ALIGNED_FREE(ref_im);
    }

    printf("ratio = MKL/ours (>1 = we're faster)\n");
    return 0;
}
