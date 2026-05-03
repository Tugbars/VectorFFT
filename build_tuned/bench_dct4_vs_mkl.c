/* bench_dct4_vs_mkl.c -- DCT-IV (REDFT11) vs MKL TT API.
 *
 * MKL TT (Trigonometric Transforms) is NOT batched — it operates on a single
 * 1D array. To process K transforms we loop K times, gathering each column
 * into a contiguous buffer first. This is the natural way to call MKL for
 * batched DCT-IV; the gather cost is part of MKL's "API tax."
 *
 * Layout: input is x[n*K + k] (samples-outer, batch-inner).
 *   For our DCT-IV: pass directly.
 *   For MKL TT: gather column k into mkl_buf, call d_forward_trig_transform,
 *               scatter back (so we can verify correctness).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "planner.h"
#include "dct4.h"
#include "env.h"

#include "mkl.h"
#include "mkl_trig_transforms.h"

static double dct4_now_ns(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}

static double max_abs_diff(const double *a, const double *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) { double d = fabs(a[i]-b[i]); if (d > m) m = d; }
    return m;
}

int main(void) {
    stride_env_init();
    stride_pin_thread(0);
    stride_set_num_threads(1);
    mkl_set_num_threads(1);
    stride_registry_t reg; stride_registry_init(&reg);
    stride_wisdom_t wis; stride_wisdom_init(&wis);
    stride_wisdom_load(&wis, "vfft_wisdom_tuned.txt");

    printf("=== bench_dct4_vs_mkl -- DCT-IV (REDFT11) ===\n");
    printf("Note: MKL TT is single-transform; we loop K times.\n\n");
    printf("N      K     vfft_ns       mkl_ns        ratio   correctness\n");
    printf("-----+-----+------------+------------+-------+------------\n");

    struct { int N; size_t K; } cells[] = {
        {  8, 256 }, {  8, 1024 }, {  8, 4096 },
        { 16, 256 }, { 16, 1024 },
        { 32, 256 }, { 32, 1024 },
        { 64, 256 }, { 64, 1024 },
        { 256, 256 }, { 1024, 256 },
    };
    int n_cells = (int)(sizeof(cells)/sizeof(cells[0]));
    const int reps = 21;

    /* MKL TT scratch sizes (generous): ipar 128 ints, dpar ~3N+10 doubles */
    MKL_INT ipar[128];

    for (int ci = 0; ci < n_cells; ci++) {
        int N = cells[ci].N; size_t K = cells[ci].K; size_t NK = (size_t)N*K;
        double *src      = (double *)mkl_malloc(NK*sizeof(double), 64);
        double *out_us   = (double *)mkl_malloc(NK*sizeof(double), 64);
        double *out_mkl  = (double *)mkl_malloc(NK*sizeof(double), 64);
        double *mkl_buf  = (double *)mkl_malloc((size_t)N*sizeof(double), 64);
        size_t dpar_sz = (size_t)(5*N/2 + 2 + 32);  /* generous */
        double *dpar    = (double *)mkl_malloc(dpar_sz*sizeof(double), 64);

        srand(42 + N + (int)K);
        for (size_t i = 0; i < NK; i++) src[i] = (double)rand()/RAND_MAX - 0.5;

        stride_plan_t *plan = stride_dct4_wise_plan(N, K, &reg, &wis);
        if (!plan) { printf("%d %zu PLAN_FAIL\n", N, K); continue; }

        /* MKL TT init: DCT-IV = MKL_STAGGERED2_COSINE_TRANSFORM */
        MKL_INT mn = N, tt_type = MKL_STAGGERED2_COSINE_TRANSFORM, ir = 0;
        DFTI_DESCRIPTOR_HANDLE handle = NULL;
        memcpy(mkl_buf, src, N*sizeof(double));
        d_init_trig_transform(&mn, &tt_type, ipar, dpar, &ir);
        if (ir != 0) { printf("init failed ir=%lld\n", (long long)ir); continue; }
        d_commit_trig_transform(mkl_buf, &handle, ipar, dpar, &ir);
        if (ir != 0) { printf("commit failed ir=%lld\n", (long long)ir); continue; }

        /* vfft bench */
        double vfft_min = 1e18;
        for (int it = 0; it < reps; it++) {
            memcpy(out_us, src, NK*sizeof(double));
            double t0 = dct4_now_ns();
            plan->override_fwd(plan->override_data, out_us, NULL);
            double t1 = dct4_now_ns();
            if (t1-t0 < vfft_min) vfft_min = t1-t0;
        }

        /* MKL bench: loop over K, gather column, transform, scatter */
        double mkl_min = 1e18;
        for (int it = 0; it < reps; it++) {
            double t0 = dct4_now_ns();
            for (size_t k = 0; k < K; k++) {
                /* gather */
                for (int n = 0; n < N; n++) mkl_buf[n] = src[(size_t)n*K + k];
                /* transform */
                d_forward_trig_transform(mkl_buf, &handle, ipar, dpar, &ir);
                /* scatter */
                for (int n = 0; n < N; n++) out_mkl[(size_t)n*K + k] = mkl_buf[n];
            }
            double t1 = dct4_now_ns();
            if (t1-t0 < mkl_min) mkl_min = t1-t0;
        }

        /* Correctness: re-run both on identical input.
         * Note: MKL's DCT-IV may have a different normalization than FFTW REDFT11.
         * REDFT11: Y[k] = 2 * sum cos(...). Some MKL conventions use sum (no 2x).
         * We compare ratios instead of raw values to detect that. */
        srand(42 + N + (int)K);
        for (size_t i = 0; i < NK; i++) src[i] = (double)rand()/RAND_MAX - 0.5;
        memcpy(out_us, src, NK*sizeof(double));
        plan->override_fwd(plan->override_data, out_us, NULL);
        for (size_t k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) mkl_buf[n] = src[(size_t)n*K + k];
            d_forward_trig_transform(mkl_buf, &handle, ipar, dpar, &ir);
            for (int n = 0; n < N; n++) out_mkl[(size_t)n*K + k] = mkl_buf[n];
        }
        /* Detect MKL's scale: ratio out_us[0] / out_mkl[0] should be constant. */
        double scale = (fabs(out_mkl[0]) > 1e-30) ? out_us[0] / out_mkl[0] : 1.0;
        for (size_t i = 0; i < NK; i++) out_mkl[i] *= scale;
        double err = max_abs_diff(out_us, out_mkl, NK);

        double ratio = mkl_min / vfft_min;
        printf("%-5d %-4zu  %10.0f   %10.0f   %5.2fx  err=%.2e (mkl_scale=%.4f) %s\n",
               N, K, vfft_min, mkl_min, ratio, err, scale,
               err < (double)N * 1e-12 ? "[match]" : "[MISMATCH]");

        free_trig_transform(&handle, ipar, &ir);
        stride_plan_destroy(plan);
        mkl_free(src); mkl_free(out_us); mkl_free(out_mkl);
        mkl_free(mkl_buf); mkl_free(dpar);
    }
    return 0;
}
