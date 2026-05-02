/* bench_2d_mt.c — measure 2D FFT MT scaling (T=1..8).
 *
 * Compares against the old-core README claim of 3.99×/3.78× at T=8 for
 * 256×256 / 1024×1024.
 *
 * For each (N1, N2):
 *   - Build wisdom-aware tiled 2D plan at T_MAX
 *   - Sweep T = 1, 2, 4, 8
 *   - Time fwd+bwd roundtrip, take min over reps
 *   - Verify byte-identical results across T
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "planner.h"
#include "fft2d.h"
#include "env.h"

#define T_MAX 8

static double mt2d_now_ns(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}

static double max_abs_diff(const double *a, const double *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static int test_cell(int N1, int N2, stride_registry_t *reg, stride_wisdom_t *wis) {
    size_t NK = (size_t)N1 * N2;

    double *re_orig = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *im_orig = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *re_ref  = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *im_ref  = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *re_w    = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *im_w    = (double *)_aligned_malloc(NK * sizeof(double), 64);

    srand(13 + N1 * 100 + N2);
    for (size_t i = 0; i < NK; i++) {
        re_orig[i] = (double)rand() / RAND_MAX - 0.5;
        im_orig[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* Plan at T_MAX so per-thread scratch is allocated for max */
    stride_set_num_threads(T_MAX);
    stride_plan_t *plan = stride_plan_2d_wise(N1, N2, reg, wis);
    if (!plan) {
        printf("  N1=%-5d N2=%-5d  plan failed\n", N1, N2);
        _aligned_free(re_orig); _aligned_free(im_orig);
        _aligned_free(re_ref);  _aligned_free(im_ref);
        _aligned_free(re_w);    _aligned_free(im_w);
        return 1;
    }

    /* T=1 reference roundtrip */
    stride_set_num_threads(1);
    memcpy(re_ref, re_orig, NK * sizeof(double));
    memcpy(im_ref, im_orig, NK * sizeof(double));
    stride_execute_fwd(plan, re_ref, im_ref);
    stride_execute_bwd(plan, re_ref, im_ref);
    double inv_NK = 1.0 / (double)NK;
    for (size_t i = 0; i < NK; i++) { re_ref[i] *= inv_NK; im_ref[i] *= inv_NK; }
    double rt_err_re = max_abs_diff(re_orig, re_ref, NK);
    double rt_err_im = max_abs_diff(im_orig, im_ref, NK);
    double rt_err = rt_err_re > rt_err_im ? rt_err_re : rt_err_im;

    /* For err_vs_T1: redo T=1 fwd+bwd into re_ref/im_ref for byte-comparison */
    memcpy(re_ref, re_orig, NK * sizeof(double));
    memcpy(im_ref, im_orig, NK * sizeof(double));
    stride_execute_fwd(plan, re_ref, im_ref);
    stride_execute_bwd(plan, re_ref, im_ref);

    printf("\n[%dx%d] (NK=%zu)  roundtrip_err=%.2e\n", N1, N2, NK, rt_err);

    int fail = 0;
    if (rt_err > 1e-10) { printf("  FAIL: roundtrip err too large at T=1\n"); fail = 1; }

    int T_values[] = {1, 2, 4, 8};
    double t1_min = 0.0;

    printf("  T   fwd_min_ns      bwd_min_ns      rt_min_ns       speedup    err_vs_T1\n");
    for (int ti = 0; ti < (int)(sizeof(T_values)/sizeof(T_values[0])); ti++) {
        int T = T_values[ti];
        stride_set_num_threads(T);

        const int reps = 5;
        double fwd_min = 1e18, bwd_min = 1e18, rt_min = 1e18;
        for (int it = 0; it < reps; it++) {
            memcpy(re_w, re_orig, NK * sizeof(double));
            memcpy(im_w, im_orig, NK * sizeof(double));
            double t0 = mt2d_now_ns();
            stride_execute_fwd(plan, re_w, im_w);
            double t1 = mt2d_now_ns();
            stride_execute_bwd(plan, re_w, im_w);
            double t2 = mt2d_now_ns();
            double fwd_ns = t1 - t0, bwd_ns = t2 - t1, rt_ns = t2 - t0;
            if (fwd_ns < fwd_min) fwd_min = fwd_ns;
            if (bwd_ns < bwd_min) bwd_min = bwd_ns;
            if (rt_ns  < rt_min)  rt_min  = rt_ns;
        }

        double err_re = max_abs_diff(re_w, re_ref, NK);
        double err_im = max_abs_diff(im_w, im_ref, NK);
        double err = err_re > err_im ? err_re : err_im;
        if (T == 1) t1_min = rt_min;
        double speedup = (t1_min > 0) ? t1_min / rt_min : 1.0;

        printf("  %-3d %12.0f    %12.0f    %12.0f    %5.2fx    %.2e\n",
               T, fwd_min, bwd_min, rt_min, speedup, err);
        if (err > 1e-10) {
            printf("    FAIL: T=%d diverges from T=1 result\n", T);
            fail = 1;
        }
    }

    stride_set_num_threads(1);
    stride_plan_destroy(plan);
    _aligned_free(re_orig); _aligned_free(im_orig);
    _aligned_free(re_ref);  _aligned_free(im_ref);
    _aligned_free(re_w);    _aligned_free(im_w);
    return fail;
}

int main(void) {
    stride_env_init();
    stride_pin_thread(0);
    stride_set_num_threads(T_MAX);

    printf("=== bench_2d_mt — 2D FFT MT scaling (T_MAX=%d) ===\n", T_MAX);

    stride_registry_t reg;
    stride_registry_init(&reg);

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);
    int loaded = stride_wisdom_load(&wis, "vfft_wisdom_tuned.txt");
    printf("wisdom : vfft_wisdom_tuned.txt (%d entries)\n", loaded < 0 ? 0 : loaded);

    struct { int N1, N2; } cells[] = {
        {  64,  64 },
        { 128, 128 },
        { 256, 256 },
        { 512, 512 },
        { 1024, 1024 },
    };
    int n = (int)(sizeof(cells)/sizeof(cells[0]));

    int fail = 0;
    for (int i = 0; i < n; i++)
        fail += test_cell(cells[i].N1, cells[i].N2, &reg, &wis);

    printf("\n=== %s ===\n", fail == 0 ? "PASS" : "FAIL");
    return fail;
}
