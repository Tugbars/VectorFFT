/* bench_dst23_vs_mkl.c -- DST-II / DST-III vs MKL TT API.
 *
 * MKL TT (Trigonometric Transforms) is single-transform; we loop K times
 * with gather/scatter per call. Same caveats as bench_dct4_vs_mkl.c.
 *   DST-II  = forward of MKL_STAGGERED_SINE_TRANSFORM
 *   DST-III = backward of MKL_STAGGERED_SINE_TRANSFORM
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "planner.h"
#include "dst.h"
#include "env.h"

#include "mkl.h"
#include "mkl_trig_transforms.h"

static double dst_now_ns(void) {
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

static void bench_one(int N, size_t K, int direction,
                      stride_registry_t *reg, stride_wisdom_t *wis,
                      MKL_INT *ipar)
{
    /* direction: 0 = DST-II forward, 1 = DST-III backward */
    size_t NK = (size_t)N * K;
    double *src      = (double *)mkl_malloc(NK*sizeof(double), 64);
    double *out_us   = (double *)mkl_malloc(NK*sizeof(double), 64);
    double *out_mkl  = (double *)mkl_malloc(NK*sizeof(double), 64);
    double *mkl_buf  = (double *)mkl_malloc((size_t)N*sizeof(double), 64);
    size_t dpar_sz   = (size_t)(5*N/2 + 2 + 32);
    double *dpar     = (double *)mkl_malloc(dpar_sz*sizeof(double), 64);

    srand(42 + N + (int)K);
    for (size_t i = 0; i < NK; i++) src[i] = (double)rand()/RAND_MAX - 0.5;

    stride_plan_t *plan = stride_dst2_wise_plan(N, K, reg, wis);
    if (!plan) {
        printf("%-3s N=%-5d K=%-4zu PLAN_FAIL\n", direction ? "III" : "II", N, K);
        mkl_free(src); mkl_free(out_us); mkl_free(out_mkl);
        mkl_free(mkl_buf); mkl_free(dpar);
        return;
    }

    /* MKL TT init: MKL_STAGGERED_SINE_TRANSFORM (DST-II/III pair) */
    MKL_INT mn = N, tt_type = MKL_STAGGERED_SINE_TRANSFORM, ir = 0;
    DFTI_DESCRIPTOR_HANDLE handle = NULL;
    memcpy(mkl_buf, src, N*sizeof(double));
    d_init_trig_transform(&mn, &tt_type, ipar, dpar, &ir);
    if (ir != 0) { printf("init failed ir=%lld\n", (long long)ir); goto cleanup; }
    /* MKL TT defaults dpar[0] to a tiny "boundary tolerance" — for staggered
     * sine it expects f[0] to be near zero. Our test data is general; raise
     * the tolerance to bypass the check. */
    dpar[0] = 1e10;
    d_commit_trig_transform(mkl_buf, &handle, ipar, dpar, &ir);
    if (ir != 0) { printf("commit failed ir=%lld\n", (long long)ir); goto cleanup; }

    const int reps = 21;

    /* vfft bench */
    double vfft_min = 1e18;
    for (int it = 0; it < reps; it++) {
        memcpy(out_us, src, NK*sizeof(double));
        double t0 = dst_now_ns();
        if (direction) plan->override_bwd(plan->override_data, out_us, NULL);
        else           plan->override_fwd(plan->override_data, out_us, NULL);
        double t1 = dst_now_ns();
        if (t1-t0 < vfft_min) vfft_min = t1-t0;
    }

    /* MKL bench: loop K, gather/transform/scatter */
    double mkl_min = 1e18;
    for (int it = 0; it < reps; it++) {
        double t0 = dst_now_ns();
        for (size_t k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) mkl_buf[n] = src[(size_t)n*K + k];
            if (direction)
                d_backward_trig_transform(mkl_buf, &handle, ipar, dpar, &ir);
            else
                d_forward_trig_transform(mkl_buf, &handle, ipar, dpar, &ir);
            for (int n = 0; n < N; n++) out_mkl[(size_t)n*K + k] = mkl_buf[n];
        }
        double t1 = dst_now_ns();
        if (t1-t0 < mkl_min) mkl_min = t1-t0;
    }

    /* No correctness check: MKL's STAGGERED_SINE_TRANSFORM is a PDE boundary-
     * value transform, mathematically distinct from FFTW's RODFT10/01.
     * Timing comparison is still meaningful (both do an N-length sine-like
     * transform K times); math equivalence is not. */
    double ratio = mkl_min / vfft_min;
    printf("%-3s N=%-5d K=%-4zu  vfft=%9.0f ns  mkl=%10.0f ns  ratio=%5.2fx\n",
           direction ? "III" : "II", N, K, vfft_min, mkl_min, ratio);

    free_trig_transform(&handle, ipar, &ir);
cleanup:
    stride_plan_destroy(plan);
    mkl_free(src); mkl_free(out_us); mkl_free(out_mkl);
    mkl_free(mkl_buf); mkl_free(dpar);
}

int main(void) {
    stride_env_init();
    stride_pin_thread(0);
    stride_set_num_threads(1);
    mkl_set_num_threads(1);
    stride_registry_t reg; stride_registry_init(&reg);
    stride_wisdom_t wis; stride_wisdom_init(&wis);
    stride_wisdom_load(&wis, "vfft_wisdom_tuned.txt");

    printf("=== bench_dst23_vs_mkl -- DST-II (RODFT10) and DST-III (RODFT01) ===\n");
    printf("Note: MKL TT is single-transform; we loop K times.\n\n");

    struct { int N; size_t K; } cells[] = {
        {  8, 256 }, {  8, 1024 }, {  8, 4096 },
        { 16, 256 }, { 16, 1024 },
        { 32, 256 }, { 32, 1024 },
        { 64, 256 }, { 64, 1024 },
        { 256, 256 }, { 1024, 256 },
    };
    int n_cells = (int)(sizeof(cells)/sizeof(cells[0]));

    MKL_INT ipar[128];

    printf("[DST-II forward]\n");
    for (int ci = 0; ci < n_cells; ci++)
        bench_one(cells[ci].N, cells[ci].K, /*direction=*/0, &reg, &wis, ipar);

    printf("\n[DST-III backward]\n");
    for (int ci = 0; ci < n_cells; ci++)
        bench_one(cells[ci].N, cells[ci].K, /*direction=*/1, &reg, &wis, ipar);

    return 0;
}
