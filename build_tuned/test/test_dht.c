/* test_dht.c -- DHT (Discrete Hartley Transform) validation.
 *
 * Identity: DHT(DHT(x))/N == x  (FFTW unnormalized self-inverse).
 * Direct check for small N: H[k] = sum_n x[n] * (cos(2*pi*k*n/N) + sin(2*pi*k*n/N))
 *
 * If built with --fftw, also compares output against FFTW_DHT.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "planner.h"
#include "dht.h"
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

/* Reference: H[k] = sum_n x[n] * (cos(2*pi*k*n/N) + sin(2*pi*k*n/N)) */
static void direct_dht(const double *x, int Nlen, double *H) {
    for (int k = 0; k < Nlen; k++) {
        double s = 0.0;
        for (int n = 0; n < Nlen; n++) {
            double th = 2.0 * M_PI * (double)k * (double)n / (double)Nlen;
            s += x[n] * (cos(th) + sin(th));
        }
        H[k] = s;
    }
}

static int test_cell(int N, size_t K, stride_registry_t *reg, stride_wisdom_t *wis,
                     int do_direct) {
    size_t NK = (size_t)N * K;
    double *orig = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *Hbuf = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *back = (double *)_aligned_malloc(NK * sizeof(double), 64);

    srand(11 + N + (int)K);
    for (size_t i = 0; i < NK; i++) orig[i] = (double)rand() / RAND_MAX - 0.5;

    stride_plan_t *plan = stride_dht_wise_plan(N, K, reg, wis);
    if (!plan) {
        printf("  N=%-5d K=%-3zu  PLAN_FAIL\n", N, K);
        _aligned_free(orig); _aligned_free(Hbuf); _aligned_free(back);
        return 1;
    }

    /* Forward DHT */
    stride_execute_dht(plan, orig, Hbuf);

    /* Apply DHT a second time and divide by N to recover orig */
    stride_execute_dht(plan, Hbuf, back);
    double inv_N = 1.0 / (double)N;
    for (size_t i = 0; i < NK; i++) back[i] *= inv_N;

    double rt_err = max_abs_diff(orig, back, NK);

    /* Direct reference for small N (column k=0 only) */
    double direct_err = -1.0;
    if (do_direct) {
        double *xs  = (double *)malloc((size_t)N * sizeof(double));
        double *ref = (double *)malloc((size_t)N * sizeof(double));
        for (int n = 0; n < N; n++) xs[n] = orig[(size_t)n * K + 0];
        direct_dht(xs, N, ref);
        double max_e = 0;
        for (int k = 0; k < N; k++) {
            double d = fabs(Hbuf[(size_t)k * K + 0] - ref[k]);
            if (d > max_e) max_e = d;
        }
        direct_err = max_e;
        free(xs); free(ref);
    }

#ifdef VFFT_HAS_FFTW
    /* FFTW comparison */
    double fftw_err = -1.0;
    {
        double *finp = (double *)fftw_malloc(NK * sizeof(double));
        double *fout = (double *)fftw_malloc(NK * sizeof(double));
        memcpy(finp, orig, NK * sizeof(double));
        fftw_r2r_kind kind = FFTW_DHT;
        int n_arr[1] = { N };
        fftw_plan fp = fftw_plan_many_r2r(1, n_arr, (int)K,
            finp, NULL, (int)K, 1, fout, NULL, (int)K, 1,
            &kind, FFTW_ESTIMATE);
        memcpy(finp, orig, NK * sizeof(double));
        fftw_execute(fp);
        fftw_err = max_abs_diff(Hbuf, fout, NK);
        fftw_destroy_plan(fp);
        fftw_free(finp); fftw_free(fout);
    }
#endif

    int fail_rt = (rt_err > 1e-10);
    double dthresh = (double)N * 1e-13;
    int fail_d = (do_direct && direct_err > dthresh);
#ifdef VFFT_HAS_FFTW
    int fail_f = (fftw_err > dthresh);
#else
    int fail_f = 0;
#endif

    printf("  N=%-5d K=%-3zu  rt=%.2e", N, K, rt_err);
    if (do_direct) printf("  direct=%.2e", direct_err);
#ifdef VFFT_HAS_FFTW
    printf("  fftw=%.2e", fftw_err);
#endif
    printf("  %s\n", (fail_rt || fail_d || fail_f) ? "FAIL" : "PASS");

    stride_plan_destroy(plan);
    _aligned_free(orig); _aligned_free(Hbuf); _aligned_free(back);
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

    printf("=== test_dht -- DHT roundtrip + direct + FFTW ===\n\n");

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
