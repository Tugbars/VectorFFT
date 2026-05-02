/* bench_mt_overrides.c — validate MT scaling for Bluestein/Rader overrides.
 *
 * Sets T=T_MAX BEFORE plan creation (so n_threads is sized for the max),
 * then sweeps T=1,2,4,8 measuring roundtrip wall-clock and verifying
 * that all T agree to numerical precision against the T=1 reference.
 *
 * Cells exercised:
 *   N=311  K=256  Bluestein (4 blocks of B=64)
 *   N=509  K=256  Bluestein (4 blocks of B=64)
 *   N=2801 K=256  Rader     (1 block of B=256 — single-threaded by design,
 *                            included as a "MT shouldn't crash" smoke check)
 *   N=127  K=256  Rader     (depends on B; should split if B<K)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "planner.h"
#include "env.h"
#include "wisdom_bridge.h"
#include "bluestein.h"
#include "rader.h"

#define T_MAX 8

static double mt_now_ns(void) {
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

static void fill_random(double *re, double *im, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }
}

typedef struct {
    int N;
    size_t K;
    const char *label;
} bench_cell_t;

static int test_cell(stride_registry_t *reg, stride_wisdom_t *wis,
                     int N, size_t K, const char *label) {
    size_t NK = (size_t)N * K;
    double *re_orig = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *im_orig = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *re_ref  = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *im_ref  = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *re_w    = (double *)_aligned_malloc(NK * sizeof(double), 64);
    double *im_w    = (double *)_aligned_malloc(NK * sizeof(double), 64);

    fill_random(re_orig, im_orig, NK, 42 + N);

    /* Reset T=T_MAX before plan creation so the plan snapshots the max
     * for its scratch allocation. The previous test_cell may have ended
     * with T=1 (single-thread reference); without this reset, every plan
     * after the first would be created with n_threads=1. */
    stride_set_num_threads(T_MAX);

    stride_plan_t *plan = stride_wise_plan(N, K, reg, wis);
    if (!plan) {
        /* Wisdom miss — try auto plan */
        plan = stride_auto_plan_wis(N, K, reg, wis);
    }
    if (!plan) {
        printf("  [%s] N=%d K=%zu  FAIL: could not build plan\n", label, N, K);
        _aligned_free(re_orig); _aligned_free(im_orig);
        _aligned_free(re_ref);  _aligned_free(im_ref);
        _aligned_free(re_w);    _aligned_free(im_w);
        return 1;
    }

    /* T=1 reference: roundtrip + scaling. Result stored in re_ref/im_ref. */
    stride_set_num_threads(1);
    memcpy(re_ref, re_orig, NK * sizeof(double));
    memcpy(im_ref, im_orig, NK * sizeof(double));
    stride_execute_fwd(plan, re_ref, im_ref);
    stride_execute_bwd(plan, re_ref, im_ref);
    /* Roundtrip is unnormalized: bwd(fwd(x)) = N*x. Divide for comparison. */
    double inv_N = 1.0 / (double)N;
    for (size_t i = 0; i < NK; i++) { re_ref[i] *= inv_N; im_ref[i] *= inv_N; }

    /* Roundtrip error vs original (sanity) */
    double rt_err = max_abs_diff(re_orig, re_ref, NK);
    double rt_err_im = max_abs_diff(im_orig, im_ref, NK);
    if (rt_err_im > rt_err) rt_err = rt_err_im;

    /* Peek at override block-size info if applicable */
    char info[128] = "";
    if (plan->override_fwd == _bluestein_execute_fwd) {
        stride_bluestein_data_t *bd = (stride_bluestein_data_t *)plan->override_data;
        snprintf(info, sizeof(info), " M=%d B=%zu n_blocks=%zu n_threads=%d",
                 bd->M, bd->B, (K + bd->B - 1) / bd->B, bd->n_threads);
    } else if (plan->override_fwd == _rader_execute_fwd) {
        stride_rader_data_t *rd = (stride_rader_data_t *)plan->override_data;
        snprintf(info, sizeof(info), " B=%zu n_blocks=%zu n_threads=%d",
                 rd->B, (K + rd->B - 1) / rd->B, rd->n_threads);
    }

    printf("\n[%s] N=%d K=%zu  roundtrip_err=%.2e%s\n", label, N, K, rt_err, info);

    int fail = 0;
    if (rt_err > 1e-10) {
        printf("  FAIL: roundtrip error too large at T=1\n");
        fail = 1;
    }

    /* Bench at T=1 first, again, to get a wall-clock baseline matching ref.
     * Then sweep T=2,4,8 and compare results. */
    int T_values[] = {1, 2, 4, 8};
    double t1_min_ns = 0.0;

    printf("  T   fwd_min_ns      bwd_min_ns      rt_min_ns       speedup    err_vs_T1\n");
    for (int ti = 0; ti < (int)(sizeof(T_values)/sizeof(T_values[0])); ti++) {
        int T = T_values[ti];
        stride_set_num_threads(T);

        double fwd_min = 1e18, bwd_min = 1e18, rt_min = 1e18;
        const int iters = 5;
        for (int it = 0; it < iters; it++) {
            memcpy(re_w, re_orig, NK * sizeof(double));
            memcpy(im_w, im_orig, NK * sizeof(double));

            double t0 = now_ns();
            stride_execute_fwd(plan, re_w, im_w);
            double t1 = now_ns();
            stride_execute_bwd(plan, re_w, im_w);
            double t2 = now_ns();

            double fwd_ns = t1 - t0;
            double bwd_ns = t2 - t1;
            double rt_ns  = t2 - t0;
            if (fwd_ns < fwd_min) fwd_min = fwd_ns;
            if (bwd_ns < bwd_min) bwd_min = bwd_ns;
            if (rt_ns  < rt_min)  rt_min  = rt_ns;
        }

        /* Apply 1/N scaling to one final result for comparison */
        for (size_t i = 0; i < NK; i++) { re_w[i] *= inv_N; im_w[i] *= inv_N; }

        double err = max_abs_diff(re_w, re_ref, NK);
        double err_im = max_abs_diff(im_w, im_ref, NK);
        if (err_im > err) err = err_im;

        if (T == 1) t1_min_ns = rt_min;
        double speedup = (t1_min_ns > 0) ? t1_min_ns / rt_min : 1.0;

        printf("  %-3d %12.0f    %12.0f    %12.0f    %5.2fx    %.2e\n",
               T, fwd_min, bwd_min, rt_min, speedup, err);

        if (err > 1e-10) {
            printf("    FAIL: T=%d result diverges from T=1 reference\n", T);
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

int main(int argc, char **argv) {
    stride_env_init();

    /* Set T=T_MAX BEFORE any plans are created so they get sized for the max. */
    stride_set_num_threads(T_MAX);
    printf("=== bench_mt_overrides — validate MT for Bluestein/Rader ===\n");
    printf("T_MAX = %d (pool created before plan creation)\n", T_MAX);

    stride_registry_t reg;
    stride_registry_init(&reg);

    stride_wisdom_t wis;
    stride_wisdom_init(&wis);

    const char *wisdom_path = "vfft_wisdom_tuned.txt";
    if (argc > 1) wisdom_path = argv[1];
    int loaded = stride_wisdom_load(&wis, wisdom_path);
    printf("wisdom : %s (%d entries)\n", wisdom_path, loaded < 0 ? 0 : loaded);

    bench_cell_t cells[] = {
        /* Override-path cells (Bluestein/Rader) */
        { 311,  256, "bluestein 311:256" },
        { 509,  256, "bluestein 509:256" },
        { 127,  256, "rader     127:256" },
        { 2801, 256, "rader    2801:256" },
        /* Direct executor path (K-split / group-parallel) — for comparison */
        { 1024, 256, "pow2      1024:256" },
        { 4096, 256, "pow2      4096:256" },
        { 16384, 256, "pow2     16384:256" },
        { 1000, 256, "comp      1000:256" },
    };
    int nf = 0;
    for (size_t i = 0; i < sizeof(cells)/sizeof(cells[0]); i++) {
        nf += test_cell(&reg, &wis, cells[i].N, cells[i].K, cells[i].label);
    }

    printf("\n=== %s: %d cells %s ===\n",
           nf == 0 ? "PASS" : "FAIL",
           (int)(sizeof(cells)/sizeof(cells[0])),
           nf == 0 ? "all good" : "had failures");
    return nf;
}
