/* test_dct23_rt.c — DCT-II / DCT-III roundtrip validation.
 *
 * Identity: DCT-III(DCT-II(x)) = 2N * x  (FFTW unnormalized convention).
 * Test: roundtrip every cell and verify max error after dividing by 2N.
 *
 * Also: direct-DCT-III check against reference for small N.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "dct.h"
#include "env.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double max_abs_diff(const double *a, const double *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

/* Reference DCT-III (FFTW REDFT01):
 *   Y[k] = X[0] + 2 * sum_{n=1..N-1} X[n] * cos(πn(2k+1)/(2N)) */
static void direct_dct3(const double *X, int Nlen, double *Y) {
    for (int k = 0; k < Nlen; k++) {
        double s = X[0];
        for (int n = 1; n < Nlen; n++)
            s += 2.0 * X[n] * cos(M_PI * n * (2*k + 1) / (2.0 * Nlen));
        Y[k] = s;
    }
}

static int test_cell(int N, size_t K, stride_registry_t *reg, stride_wisdom_t *wis,
                     int do_direct) {
    size_t NK = (size_t)N * K;
    double *orig  = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *coefs = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *back  = (double *)_aligned_malloc(NK * sizeof(double), 64);

    srand(7 + N + (int)K);
    for (size_t i = 0; i < NK; i++) orig[i] = (double)rand() / RAND_MAX - 0.5;

    stride_plan_t *plan = stride_dct2_wise_plan(N, K, reg, wis);
    if (!plan) {
        printf("  N=%-5d K=%-3zu  PLAN_FAIL\n", N, K);
        _aligned_free(orig); _aligned_free(coefs); _aligned_free(back);
        return 1;
    }

    /* DCT-II forward */
    stride_execute_dct2(plan, orig, coefs);

    /* DCT-III backward — should give 2N * orig */
    stride_execute_dct3(plan, coefs, back);

    /* Normalize and check */
    double inv_2N = 1.0 / (2.0 * (double)N);
    for (size_t i = 0; i < NK; i++) back[i] *= inv_2N;
    double rt_err = max_abs_diff(orig, back, NK);

    /* Direct DCT-III check: take coefs (the DCT-II output), apply direct
     * reference DCT-III to it, compare to our DCT-III output. */
    double direct_err = -1.0;
    if (do_direct) {
        double *ref  = (double *)malloc(N * sizeof(double));
        double *cs   = (double *)malloc(N * sizeof(double));
        for (int n = 0; n < N; n++) cs[n] = coefs[(size_t)n * K + 0];
        direct_dct3(cs, N, ref);
        /* Re-run DCT-III on coefs (without the inv_2N normalization we did to back)
         * Actually `back` has been normalized; let's just call dct3 again. */
        double *back_un = (double *)malloc(N * K * sizeof(double));
        stride_execute_dct3(plan, coefs, back_un);
        double max_e = 0;
        for (int k = 0; k < N; k++) {
            double d = fabs(back_un[(size_t)k * K + 0] - ref[k]);
            if (d > max_e) max_e = d;
        }
        direct_err = max_e;
        free(ref); free(cs); free(back_un);
    }

    int fail_rt = (rt_err > 1e-10);
    double dthresh = (double)N * 1e-13;
    int fail_d  = (do_direct && direct_err > dthresh);

    printf("  N=%-5d K=%-3zu  rt=%.2e", N, K, rt_err);
    if (do_direct) printf("  direct=%.2e", direct_err);
    printf("  %s\n", (fail_rt || fail_d) ? "FAIL" : "PASS");

    stride_plan_destroy(plan);
    _aligned_free(orig); _aligned_free(coefs); _aligned_free(back);
    return (fail_rt || fail_d);
}

int main(void) {
    stride_env_init();
    stride_set_num_threads(1);

    stride_registry_t reg;
    stride_registry_init(&reg);

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);
    stride_wisdom_load(&wis, "vfft_wisdom_tuned.txt");

    printf("=== test_dct23_rt — DCT-II/DCT-III roundtrip + direct check ===\n\n");

    struct { int N; size_t K; } cells[] = {
        {   8, 4 }, {   8, 32 }, {   8, 1024 },   /* JPEG-block-ish */
        {  16, 4 }, {  16, 32 },
        {  32, 4 }, {  32, 32 }, {  32, 256 },
        {  64, 4 }, {  64, 32 }, {  64, 256 },
        { 128, 4 }, { 128, 32 }, { 128, 256 },
        { 256, 4 }, { 256, 32 }, { 256, 256 },
        { 1024, 4 }, { 1024, 32 }, { 1024, 256 },
        { 4096, 4 }, { 4096, 256 },
    };
    int n = (int)(sizeof(cells)/sizeof(cells[0]));

    int fail = 0;
    for (int i = 0; i < n; i++) {
        int do_direct = (cells[i].N <= 256);
        fail += test_cell(cells[i].N, cells[i].K, &reg, &wis, do_direct);
    }

    printf("\n=== %s: %d/%d cells passed ===\n",
           fail == 0 ? "PASS" : "FAIL", n - fail, n);
    return fail;
}
