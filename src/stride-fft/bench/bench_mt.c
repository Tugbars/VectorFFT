/**
 * bench_mt.c -- Multithreaded benchmark: VectorFFT vs FFTW vs MKL
 *
 * Compares all three libraries at matching thread counts.
 * Tests K-split and group-parallel strategies automatically.
 *
 * Usage:
 *   vfft_bench_mt [threads]
 *
 *   threads: number of threads (default: 8)
 *
 * Example:
 *   vfft_bench_mt 4
 *   vfft_bench_mt 8
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef VFFT_HAS_MKL
#include <mkl.h>
#include <mkl_dfti.h>
#endif

#include "../core/planner.h"
#include "../core/env.h"
#include "../core/compat.h"
#include "../core/threads.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================
 * Timing helper
 * ================================================================ */

static double bench_one(void (*fn)(void *), void *arg, double min_secs) {
    /* Warmup */
    for (int i = 0; i < 10; i++) fn(arg);

    /* Timed loop */
    double t0 = now_ns();
    long long iters = 0;
    while ((now_ns() - t0) < min_secs * 1e9) {
        fn(arg);
        iters++;
    }
    return (now_ns() - t0) / (double)iters;
}

/* ================================================================
 * VectorFFT runner
 * ================================================================ */

typedef struct {
    stride_plan_t *plan;
    double *re, *im;
} vfft_ctx_t;

static void vfft_run(void *arg) {
    vfft_ctx_t *c = (vfft_ctx_t *)arg;
    stride_execute_fwd(c->plan, c->re, c->im);
}

/* MKL is benchmarked inline (in-place, no separate runner needed) */

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv) {
    int T = (argc > 1) ? atoi(argv[1]) : 8;
    if (T < 1) T = 1;

    unsigned int saved = stride_env_init();
    stride_print_info();

    printf("\n=== Multithreaded Benchmark: %d threads ===\n\n", T);

    /* Configure threading for all libraries */
    stride_set_num_threads(T);
    if (T > 1) stride_pin_thread(0);  /* pin caller to P-core 0 */

#ifdef VFFT_HAS_MKL
    mkl_set_num_threads(T);
#endif

    /* Load wisdom for optimal factorizations */
    stride_wisdom_t wis;
    stride_wisdom_init(&wis);
#ifdef VFFT_BENCH_DIR
    int wis_loaded = (stride_wisdom_load(&wis, VFFT_BENCH_DIR "/vfft_wisdom.txt") == 0);
#else
    int wis_loaded = (stride_wisdom_load(&wis, "vfft_wisdom.txt") == 0);
#endif
    if (wis_loaded && wis.count > 0)
        printf("Wisdom: %d entries loaded\n\n", wis.count);
    else
        printf("Wisdom: not found, using heuristic plans\n\n");

    /* Test cases: same N values as single-threaded bench */
    typedef struct { int N; size_t K; } test_case_t;
    test_case_t cases[] = {
        {256,    256},
        {1024,   256},
        {4096,   256},
        {60,     256},
        {200,    256},
        {1000,   256},
        {1000,   1024},
        {5000,   256},
        {10000,  256},
        {20000,  256},
        {50000,  256},
        {49,     256},
        {143,    256},
        {875,    256},
        {2401,   256},
    };
    int ncases = sizeof(cases) / sizeof(cases[0]);

    printf("%-7s %-6s | %-20s %12s | %12s | %7s\n",
           "N", "K", "factors", "ours_ns", "mkl_ns", "vs_mkl");
    printf("-------+------+----------------------+--------------+--------------+--------\n");

    stride_registry_t reg;
    stride_registry_init(&reg);

    double bench_secs = 3.0;

    for (int ci = 0; ci < ncases; ci++) {
        int N = cases[ci].N;
        size_t K = cases[ci].K;
        size_t total = (size_t)N * K;
        size_t bytes = total * sizeof(double);

        double *re = (double *)STRIDE_ALIGNED_ALLOC(64, bytes);
        double *im = (double *)STRIDE_ALIGNED_ALLOC(64, bytes);
        if (!re || !im) { printf("ALLOC FAILED N=%d K=%zu\n", N, K); continue; }

        /* Init data */
        srand(42);
        for (size_t i = 0; i < total; i++) {
            re[i] = (double)rand() / RAND_MAX - 0.5;
            im[i] = (double)rand() / RAND_MAX - 0.5;
        }

        /* ── VectorFFT (use wisdom if available) ── */
        stride_plan_t *plan = stride_wise_plan(N, K, &reg, &wis);
        if (!plan) { printf("PLAN FAILED N=%d\n", N); continue; }

        char factors[64] = "";
        for (int s = 0; s < plan->num_stages; s++) {
            char buf[16];
            sprintf(buf, "%s%d", s ? "x" : "", plan->factors[s]);
            strcat(factors, buf);
        }

        printf("  N=%-6d K=%-6zu ours...", N, K); fflush(stdout);

        vfft_ctx_t vc = {plan, re, im};
        double ours_ns = bench_one(vfft_run, &vc, bench_secs);
        printf(" %.0f ns  ", ours_ns); fflush(stdout);

        /* ── FFTW ── */
        /* ── MKL ── */
        printf("mkl..."); fflush(stdout);
        double mkl_ns = 0;
#ifdef VFFT_HAS_MKL
        {
            DFTI_DESCRIPTOR_HANDLE h = NULL;
            MKL_LONG status = DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
            if (status == 0) {
                DftiSetValue(h, DFTI_PLACEMENT, DFTI_INPLACE);
                DftiSetValue(h, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
                MKL_LONG strides[2] = {0, (MKL_LONG)K};
                DftiSetValue(h, DFTI_INPUT_STRIDES, strides);
                DftiSetValue(h, DFTI_OUTPUT_STRIDES, strides);
                DftiSetValue(h, DFTI_INPUT_DISTANCE, 1);
                DftiSetValue(h, DFTI_OUTPUT_DISTANCE, 1);
                DftiSetValue(h, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
                status = DftiCommitDescriptor(h);
            }
            if (status == 0) {
                /* Warmup */
                for (int w = 0; w < 10; w++)
                    DftiComputeForward(h, re, im);

                double t0 = now_ns();
                long long iters = 0;
                while ((now_ns() - t0) < bench_secs * 1e9) {
                    DftiComputeForward(h, re, im);
                    iters++;
                }
                mkl_ns = (now_ns() - t0) / (double)iters;
            }
            if (h) DftiFreeDescriptor(&h);
        }
#endif

        /* Print results */
        double vs_mkl = mkl_ns > 0 ? mkl_ns / ours_ns : 0;

        printf("%-7d %-6zu | %-20s %12.1f | %12.1f | %6.2fx\n",
               N, K, factors, ours_ns, mkl_ns, vs_mkl);
        fflush(stdout);

        stride_plan_destroy(plan);
        STRIDE_ALIGNED_FREE(re);
        STRIDE_ALIGNED_FREE(im);
    }

    printf("\nThreads: %d (VectorFFT + MKL both at %d threads)\n", T, T);
    printf("Platform: single-socket, pinned to cores 0..%d\n", T - 1);

    stride_set_num_threads(1);
    stride_env_restore(saved);
    return 0;
}
