/* bench_r2c.c — measure R2C speedup vs equivalent complex FFT.
 *
 * Theoretical speedup: 2× (R2C does an N/2-point complex FFT + O(N) post-process,
 * vs the full N-point complex FFT). In practice, R2C can exceed 2× because the
 * smaller inner FFT fits cache better.
 *
 * For each (N, K):
 *   1. Build R2C plan (N-point real)
 *   2. Build complex FFT plan (N-point complex)
 *   3. Time both single-threaded, take min over reps
 *   4. Report: r2c_ns, complex_ns, speedup
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "planner.h"
#include "r2c.h"
#include "env.h"

static double r2c_now_ns(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}

typedef struct { int N; size_t K; } cell_t;

int main(void) {
    stride_env_init();
    stride_pin_thread(0);
    stride_set_num_threads(1);

    stride_registry_t reg;
    stride_registry_init(&reg);

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);
    stride_wisdom_load(&wis, "vfft_wisdom_tuned.txt");

    cell_t cells[] = {
        {   64, 4 }, {   64, 32 }, {   64, 256 },
        {  128, 4 }, {  128, 32 }, {  128, 256 },
        {  256, 4 }, {  256, 32 }, {  256, 256 },
        {  512, 4 }, {  512, 32 }, {  512, 256 },
        { 1024, 4 }, { 1024, 32 }, { 1024, 256 },
        { 2048, 4 }, { 2048, 32 }, { 2048, 256 },
        { 4096, 4 }, { 4096, 32 }, { 4096, 256 },
        { 8192, 4 }, { 8192, 32 }, { 8192, 256 },
        {16384, 4 }, {16384, 32 }, {16384, 256 },
    };
    int n_cells = (int)(sizeof(cells)/sizeof(cells[0]));

    printf("=== bench_r2c — R2C vs complex N-point FFT (single-thread) ===\n");
    printf("N        K     r2c_ns       cplx_ns      speedup    note\n");
    printf("------+-----+------------+------------+----------+------\n");

    const int reps = 21;

    for (int ci = 0; ci < n_cells; ci++) {
        int N = cells[ci].N;
        size_t K = cells[ci].K;
        size_t real_NK = (size_t)N * K;

        /* R2C plan (wisdom-aware inner) + complex plan (wisdom-aware) */
        stride_plan_t *r2c_plan = stride_r2c_wise_plan(N, K, &reg, &wis);
        stride_plan_t *cplx_plan = stride_wise_plan(N, K, &reg, &wis);
        if (!cplx_plan) cplx_plan = stride_auto_plan_wis(N, K, &reg, &wis);
        if (!r2c_plan || !cplx_plan) {
            printf("%-6d %-4zu  plan failed\n", N, K);
            if (r2c_plan)  stride_plan_destroy(r2c_plan);
            if (cplx_plan) stride_plan_destroy(cplx_plan);
            continue;
        }

        /* Buffers — both sized N*K (R2C uses real-size for in-place freq output) */
        double *r2c_re = (double *)_aligned_malloc(real_NK * sizeof(double), 64);
        double *r2c_im = (double *)_aligned_malloc(real_NK * sizeof(double), 64);
        double *cplx_re = (double *)_aligned_malloc(real_NK * sizeof(double), 64);
        double *cplx_im = (double *)_aligned_malloc(real_NK * sizeof(double), 64);
        double *src     = (double *)_aligned_malloc(real_NK * sizeof(double), 64);

        srand(42 + N + (int)K);
        for (size_t i = 0; i < real_NK; i++)
            src[i] = (double)rand() / RAND_MAX - 0.5;

        /* Bench R2C: copy src into r2c_re each rep, run override_fwd */
        double r2c_min = 1e18;
        for (int it = 0; it < reps; it++) {
            memcpy(r2c_re, src, real_NK * sizeof(double));
            double t0 = r2c_now_ns();
            r2c_plan->override_fwd(r2c_plan->override_data, r2c_re, r2c_im);
            double t1 = r2c_now_ns();
            double ns = t1 - t0;
            if (ns < r2c_min) r2c_min = ns;
        }

        /* Bench complex FFT: copy src into cplx_re, zero cplx_im, run fwd */
        double cplx_min = 1e18;
        for (int it = 0; it < reps; it++) {
            memcpy(cplx_re, src, real_NK * sizeof(double));
            memset(cplx_im, 0, real_NK * sizeof(double));
            double t0 = r2c_now_ns();
            stride_execute_fwd(cplx_plan, cplx_re, cplx_im);
            double t1 = r2c_now_ns();
            double ns = t1 - t0;
            if (ns < cplx_min) cplx_min = ns;
        }

        double speedup = cplx_min / r2c_min;
        const char *note = speedup > 2.0 ? "above 2x" : "";
        printf("%-6d %-4zu  %10.0f   %10.0f   %6.2fx    %s\n",
               N, K, r2c_min, cplx_min, speedup, note);

        _aligned_free(r2c_re); _aligned_free(r2c_im);
        _aligned_free(cplx_re); _aligned_free(cplx_im);
        _aligned_free(src);
        stride_plan_destroy(r2c_plan);
        stride_plan_destroy(cplx_plan);
    }

    return 0;
}
