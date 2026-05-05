/* bench_dct4_vs_fftw.c -- DCT-IV (REDFT11) perf vs FFTW3.
 *
 * Verifies the cost model: current DCT-IV uses 2 R2C-based transforms
 * (DCT-III + DST-III), so should be ~2x DCT-II/III latency.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "planner.h"
#include "dct4.h"
#include "env.h"
#include "fftw3.h"

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
    stride_registry_t reg; stride_registry_init(&reg);
    stride_wisdom_t wis; stride_wisdom_init(&wis);
    stride_wisdom_load(&wis, "vfft_wisdom_tuned.txt");

    printf("=== bench_dct4_vs_fftw -- DCT-IV (REDFT11) ===\n\n");
    printf("N      K     vfft_ns       fftw_ns       ratio   correctness\n");
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

    for (int ci = 0; ci < n_cells; ci++) {
        int N = cells[ci].N; size_t K = cells[ci].K; size_t NK = (size_t)N*K;
        double *src      = (double *)fftw_malloc(NK*sizeof(double));
        double *out_us   = (double *)fftw_malloc(NK*sizeof(double));
        double *out_fftw = (double *)fftw_malloc(NK*sizeof(double));
        srand(42 + N + (int)K);
        for (size_t i = 0; i < NK; i++) src[i] = (double)rand()/RAND_MAX - 0.5;

        stride_plan_t *plan = stride_dct4_wise_plan(N, K, &reg, &wis);
        if (!plan) { printf("%d %zu PLAN_FAIL\n", N, K); continue; }

        fftw_r2r_kind kind = FFTW_REDFT11;
        int n_arr[1] = { N };
        fftw_plan fp = fftw_plan_many_r2r(1, n_arr, (int)K,
            src, NULL, (int)K, 1, out_fftw, NULL, (int)K, 1,
            &kind, FFTW_MEASURE | FFTW_DESTROY_INPUT);
        srand(42 + N + (int)K);
        for (size_t i = 0; i < NK; i++) src[i] = (double)rand()/RAND_MAX - 0.5;

        double vfft_min = 1e18;
        for (int it = 0; it < reps; it++) {
            memcpy(out_us, src, NK*sizeof(double));
            double t0 = dct4_now_ns();
            plan->override_fwd(plan->override_data, out_us, NULL);
            double t1 = dct4_now_ns();
            if (t1-t0 < vfft_min) vfft_min = t1-t0;
        }
        double fftw_min = 1e18;
        for (int it = 0; it < reps; it++) {
            double t0 = dct4_now_ns();
            fftw_execute(fp);
            double t1 = dct4_now_ns();
            if (t1-t0 < fftw_min) fftw_min = t1-t0;
        }
        srand(42 + N + (int)K);
        for (size_t i = 0; i < NK; i++) src[i] = (double)rand()/RAND_MAX - 0.5;
        memcpy(out_us, src, NK*sizeof(double));
        plan->override_fwd(plan->override_data, out_us, NULL);
        fftw_execute(fp);
        double err = max_abs_diff(out_us, out_fftw, NK);
        double ratio = fftw_min / vfft_min;
        printf("%-5d %-4zu  %10.0f   %10.0f   %5.2fx  err=%.2e %s\n",
               N, K, vfft_min, fftw_min, ratio, err,
               err < (double)N * 1e-12 ? "[match]" : "[MISMATCH]");

        fftw_destroy_plan(fp); stride_plan_destroy(plan);
        fftw_free(src); fftw_free(out_us); fftw_free(out_fftw);
    }
    fftw_cleanup();
    return 0;
}
