/* test_dst23_rt.c -- DST-II / DST-III roundtrip + FFTW comparison.
 *
 * Identity: DST-III(DST-II(x))/(2N) == x  (FFTW unnormalized convention).
 * Direct check for small N against textbook DST-II.
 * If --fftw, compare against FFTW RODFT10 / RODFT01.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "dst.h"
#include "env.h"

#ifdef VFFT_HAS_FFTW
#include "fftw3.h"
#endif

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

/* Reference DST-II: Y[k] = 2 sum_{n=0..N-1} x[n] sin(pi*(k+1)*(2n+1)/(2N)) */
static void direct_dst2(const double *x, int Nlen, double *Y) {
    for (int k = 0; k < Nlen; k++) {
        double s = 0.0;
        for (int n = 0; n < Nlen; n++)
            s += 2.0 * x[n] * sin(M_PI * (k + 1) * (2*n + 1) / (2.0 * Nlen));
        Y[k] = s;
    }
}

static int test_cell(int N, size_t K, stride_registry_t *reg, stride_wisdom_t *wis,
                     int do_direct) {
    size_t NK = (size_t)N * K;
    double *orig  = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *coefs = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *back  = (double *)_aligned_malloc(NK * sizeof(double), 64);

    srand(13 + N + (int)K);
    for (size_t i = 0; i < NK; i++) orig[i] = (double)rand() / RAND_MAX - 0.5;

    stride_plan_t *plan = stride_dst2_wise_plan(N, K, reg, wis);
    if (!plan) {
        printf("  N=%-5d K=%-3zu  PLAN_FAIL\n", N, K);
        _aligned_free(orig); _aligned_free(coefs); _aligned_free(back);
        return 1;
    }

    /* DST-II forward */
    stride_execute_dst2(plan, orig, coefs);

    /* DST-III backward — should give 2N * orig */
    stride_execute_dst3(plan, coefs, back);
    double inv_2N = 1.0 / (2.0 * (double)N);
    for (size_t i = 0; i < NK; i++) back[i] *= inv_2N;
    double rt_err = max_abs_diff(orig, back, NK);

    /* Direct reference (column k=0) for small N */
    double direct_err = -1.0;
    if (do_direct) {
        double *xs  = (double *)malloc((size_t)N * sizeof(double));
        double *ref = (double *)malloc((size_t)N * sizeof(double));
        for (int n = 0; n < N; n++) xs[n] = orig[(size_t)n * K + 0];
        direct_dst2(xs, N, ref);
        double max_e = 0;
        for (int k = 0; k < N; k++) {
            double d = fabs(coefs[(size_t)k * K + 0] - ref[k]);
            if (d > max_e) max_e = d;
        }
        direct_err = max_e;
        free(xs); free(ref);
    }

#ifdef VFFT_HAS_FFTW
    double fftw_dst2_err = -1.0, fftw_dst3_err = -1.0;
    {
        double *finp = (double *)fftw_malloc(NK * sizeof(double));
        double *fout = (double *)fftw_malloc(NK * sizeof(double));

        /* DST-II via FFTW RODFT10 */
        memcpy(finp, orig, NK * sizeof(double));
        fftw_r2r_kind k2 = FFTW_RODFT10;
        int n_arr[1] = { N };
        fftw_plan fp2 = fftw_plan_many_r2r(1, n_arr, (int)K,
            finp, NULL, (int)K, 1, fout, NULL, (int)K, 1,
            &k2, FFTW_ESTIMATE);
        memcpy(finp, orig, NK * sizeof(double));
        fftw_execute(fp2);
        fftw_dst2_err = max_abs_diff(coefs, fout, NK);
        fftw_destroy_plan(fp2);

        /* DST-III via FFTW RODFT01 (apply to coefs) */
        memcpy(finp, coefs, NK * sizeof(double));
        fftw_r2r_kind k3 = FFTW_RODFT01;
        fftw_plan fp3 = fftw_plan_many_r2r(1, n_arr, (int)K,
            finp, NULL, (int)K, 1, fout, NULL, (int)K, 1,
            &k3, FFTW_ESTIMATE);
        memcpy(finp, coefs, NK * sizeof(double));
        fftw_execute(fp3);
        /* compare against our DST-III applied to coefs (call it again) */
        double *our3 = (double *)malloc(NK * sizeof(double));
        stride_execute_dst3(plan, coefs, our3);
        fftw_dst3_err = max_abs_diff(our3, fout, NK);
        free(our3);
        fftw_destroy_plan(fp3);

        fftw_free(finp); fftw_free(fout);
    }
#endif

    int fail_rt = (rt_err > 1e-10);
    double dthresh = (double)N * 1e-13;
    int fail_d = (do_direct && direct_err > dthresh);
#ifdef VFFT_HAS_FFTW
    int fail_f = (fftw_dst2_err > dthresh) || (fftw_dst3_err > dthresh);
#else
    int fail_f = 0;
#endif

    printf("  N=%-5d K=%-3zu  rt=%.2e", N, K, rt_err);
    if (do_direct) printf("  direct=%.2e", direct_err);
#ifdef VFFT_HAS_FFTW
    printf("  fftw_dst2=%.2e fftw_dst3=%.2e", fftw_dst2_err, fftw_dst3_err);
#endif
    printf("  %s\n", (fail_rt || fail_d || fail_f) ? "FAIL" : "PASS");

    stride_plan_destroy(plan);
    _aligned_free(orig); _aligned_free(coefs); _aligned_free(back);
    return (fail_rt || fail_d || fail_f);
}

int main(void) {
    stride_env_init();
    stride_set_num_threads(1);

    stride_registry_t reg;
    stride_registry_init(&reg);

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);
    stride_wisdom_load(&wis, "vfft_wisdom_tuned.txt");

    printf("=== test_dst23_rt -- DST-II/III roundtrip + direct + FFTW ===\n\n");

    struct { int N; size_t K; } cells[] = {
        {   8,   4 }, {   8,  32 }, {   8, 1024 },
        {  16,   4 }, {  16,  32 },
        {  32,   4 }, {  32,  32 }, {  32, 256 },
        {  64,   4 }, {  64,  32 }, {  64, 256 },
        { 128,   4 }, { 128,  32 }, { 128, 256 },
        { 256,   4 }, { 256,  32 }, { 256, 256 },
        { 1024,  4 }, { 1024, 32 }, { 1024, 256 },
        { 4096,  4 }, { 4096, 256 },
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
