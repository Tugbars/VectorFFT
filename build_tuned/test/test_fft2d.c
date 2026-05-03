/* test_fft2d.c — validate 2D FFT (tiled + Bailey) in new core.
 *
 * For each (N1, N2):
 *   1. Build tiled and Bailey plans
 *   2. Random complex input
 *   3. Roundtrip: fwd then bwd, divide by N1*N2, compare to original
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "fft2d.h"
#include "env.h"

static double max_abs_diff(const double *a, const double *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static int test_2d_cell(int N1, int N2, stride_registry_t *reg, stride_wisdom_t *wis,
                        int use_bailey) {
    size_t NK = (size_t)N1 * N2;
    double *re = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *im = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *re_orig = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *im_orig = (double *)_aligned_malloc(NK * sizeof(double), 64);

    srand(13 + N1 * 100 + N2);
    for (size_t i = 0; i < NK; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }
    memcpy(re_orig, re, NK * sizeof(double));
    memcpy(im_orig, im, NK * sizeof(double));

    stride_plan_t *plan = use_bailey
        ? stride_plan_2d_bailey(N1, N2, reg)
        : stride_plan_2d_wise(N1, N2, reg, wis);
    if (!plan) {
        printf("  %-7s N1=%-5d N2=%-5d  plan failed\n",
               use_bailey ? "bailey" : "tiled", N1, N2);
        _aligned_free(re); _aligned_free(im);
        _aligned_free(re_orig); _aligned_free(im_orig);
        return 1;
    }

    stride_execute_fwd(plan, re, im);
    stride_execute_bwd(plan, re, im);

    double inv_NK = 1.0 / (double)NK;
    for (size_t i = 0; i < NK; i++) { re[i] *= inv_NK; im[i] *= inv_NK; }

    double err_re = max_abs_diff(re, re_orig, NK);
    double err_im = max_abs_diff(im, im_orig, NK);
    double err = err_re > err_im ? err_re : err_im;
    int fail = (err > 1e-10) ? 1 : 0;

    printf("  %-7s N1=%-5d N2=%-5d  err=%.2e  %s\n",
           use_bailey ? "bailey" : "tiled", N1, N2, err, fail ? "FAIL" : "PASS");

    stride_plan_destroy(plan);
    _aligned_free(re); _aligned_free(im);
    _aligned_free(re_orig); _aligned_free(im_orig);
    return fail;
}

int main(void) {
    stride_env_init();
    stride_set_num_threads(1);

    stride_registry_t reg;
    stride_registry_init(&reg);

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);
    stride_wisdom_load(&wis, "vfft_wisdom_tuned.txt");

    printf("=== test_fft2d — validate 2D FFT in new core ===\n");

    struct { int N1, N2; } cells[] = {
        {  16,  16 }, {  32,  32 }, {  64,  64 },
        { 128, 128 }, { 256, 256 }, { 512, 512 }, { 1024, 1024 },
        {  64, 256 }, { 256,  64 },     /* asymmetric */
    };
    int n = (int)(sizeof(cells)/sizeof(cells[0]));

    int fail = 0;
    for (int i = 0; i < n; i++) {
        fail += test_2d_cell(cells[i].N1, cells[i].N2, &reg, &wis, /*bailey=*/0);
        fail += test_2d_cell(cells[i].N1, cells[i].N2, &reg, &wis, /*bailey=*/1);
    }

    printf("\n=== %s: %d/%d cells passed ===\n",
           fail == 0 ? "PASS" : "FAIL", n*2 - fail, n*2);
    return fail;
}
