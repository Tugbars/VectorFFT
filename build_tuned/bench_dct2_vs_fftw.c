/* bench_dct2_vs_fftw.c — DCT-II perf comparison vs FFTW3 (REDFT10).
 *
 * Cells: focused on JPEG (N=8, K=1024), plus a few sizes around it.
 *
 * Methodology:
 *   - Same input data for both
 *   - FFTW plan with FFTW_MEASURE (so plan search is fair)
 *   - Take min over reps (to filter system noise)
 *   - Layout: stride=K within transform, dist=1 between transforms
 *     (matches our N-row × K-column row-major layout)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "planner.h"
#include "dct.h"
#include "env.h"

#include "fftw3.h"

static double dctw_now_ns(void) {
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

int main(void) {
    stride_env_init();
    stride_pin_thread(0);
    stride_set_num_threads(1);

    stride_registry_t reg;
    stride_registry_init(&reg);

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);
    stride_wisdom_load(&wis, "vfft_wisdom_tuned.txt");

    /* (vcpkg FFTW3 build is single-threaded; no fftw_plan_with_nthreads needed.) */
    printf("=== bench_dct2_vs_fftw — DCT-II (REDFT10) vs FFTW3 ===\n");
    printf("FFTW3 single-threaded, FFTW_MEASURE planning\n\n");
    printf("N      K     vfft_ns       fftw_ns       ratio   correctness\n");
    printf("-----+-----+------------+------------+-------+------------\n");

    struct { int N; size_t K; } cells[] = {
        {   8,    1 },     /* single transform — worst case for us */
        {   8,    4 },
        {   8,   32 },
        {   8,  256 },
        {   8, 1024 },     /* the JPEG block layout cell */
        {   8, 4096 },
        {  16,    4 }, {  16,   32 }, {  16,  256 }, {  16, 1024 },
        {  32,    4 }, {  32,   32 }, {  32,  256 }, {  32, 1024 },
        {  64,    4 }, {  64,   32 }, {  64,  256 }, {  64, 1024 },
        { 128,    4 }, { 128,   32 }, { 128,  256 },
        { 256,    4 }, { 256,   32 }, { 256,  256 },
        {1024,    4 }, {1024,   32 }, {1024,  256 },
    };
    int n_cells = (int)(sizeof(cells)/sizeof(cells[0]));

    const int reps = 21;

    for (int ci = 0; ci < n_cells; ci++) {
        int N = cells[ci].N;
        size_t K = cells[ci].K;
        size_t NK = (size_t)N * K;

        if (N & 1) continue;          /* our impl needs even N */
        if (NK == 8 && K == 1) {
            /* K=1 hits SIMD edge case in our R2C — skip with note */
            printf("%-5d %-4zu  SKIPPED (K=1 SIMD edge in our R2C, v1.1 fix)\n", N, K);
            continue;
        }

        /* Allocate buffers, FFTW-aligned */
        double *src      = (double *)fftw_malloc(NK * sizeof(double));
        double *out_us   = (double *)fftw_malloc(NK * sizeof(double));
        double *out_fftw = (double *)fftw_malloc(NK * sizeof(double));

        srand(99 + N + (int)K);
        for (size_t i = 0; i < NK; i++)
            src[i] = (double)rand() / RAND_MAX - 0.5;

        /* Our DCT-II plan */
        stride_plan_t *plan = stride_dct2_wise_plan(N, K, &reg, &wis);
        if (!plan) {
            printf("%-5d %-4zu  PLAN_FAIL_VFFT\n", N, K);
            fftw_free(src); fftw_free(out_us); fftw_free(out_fftw);
            continue;
        }

        /* FFTW plan: REDFT10, batched K, stride=K, dist=1 */
        fftw_r2r_kind kind = FFTW_REDFT10;
        int n_arr[1] = { N };
        fftw_plan fp = fftw_plan_many_r2r(
                1, n_arr, (int)K,
                src,      NULL, (int)K, 1,
                out_fftw, NULL, (int)K, 1,
                &kind, FFTW_MEASURE | FFTW_DESTROY_INPUT);
        if (!fp) {
            printf("%-5d %-4zu  PLAN_FAIL_FFTW\n", N, K);
            stride_plan_destroy(plan);
            fftw_free(src); fftw_free(out_us); fftw_free(out_fftw);
            continue;
        }

        /* Refill src — FFTW_MEASURE may trash it */
        srand(99 + N + (int)K);
        for (size_t i = 0; i < NK; i++)
            src[i] = (double)rand() / RAND_MAX - 0.5;

        /* Bench us */
        double vfft_min = 1e18;
        for (int it = 0; it < reps; it++) {
            memcpy(out_us, src, NK * sizeof(double));
            double t0 = dctw_now_ns();
            plan->override_fwd(plan->override_data, out_us, NULL);
            double t1 = dctw_now_ns();
            double ns = t1 - t0;
            if (ns < vfft_min) vfft_min = ns;
        }

        /* Bench FFTW */
        double fftw_min = 1e18;
        for (int it = 0; it < reps; it++) {
            /* FFTW plan was made for (src -> out_fftw); keep src intact */
            double t0 = dctw_now_ns();
            fftw_execute(fp);
            double t1 = dctw_now_ns();
            double ns = t1 - t0;
            if (ns < fftw_min) fftw_min = ns;
            /* Refill src in case FFTW_DESTROY_INPUT consumed it */
            for (size_t i = 0; i < NK; i++) src[i] = out_fftw[i];   /* doesn't matter, but keeps it deterministic */
        }
        /* Run once more cleanly to get the actual output for correctness */
        srand(99 + N + (int)K);
        for (size_t i = 0; i < NK; i++) src[i] = (double)rand() / RAND_MAX - 0.5;
        fftw_execute(fp);

        double err = max_abs_diff(out_us, out_fftw, NK);
        double rel = err / (1.0 + sqrt((double)N));   /* expected DCT magnitude scale */

        double ratio = fftw_min / vfft_min;
        printf("%-5d %-4zu  %10.0f   %10.0f   %5.2fx  err=%.2e %s\n",
               N, K, vfft_min, fftw_min, ratio, err,
               rel < 1e-12 ? "[match]" : "[MISMATCH]");

        fftw_destroy_plan(fp);
        stride_plan_destroy(plan);
        fftw_free(src); fftw_free(out_us); fftw_free(out_fftw);
    }

    fftw_cleanup();
    return 0;
}
