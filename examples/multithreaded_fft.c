/*
 * multithreaded_fft.c -- VectorFFT multithreaded example
 *
 * Demonstrates the threading API and two parallelization strategies:
 *
 *   Threading (threads.h)
 *   ---------------------
 *   stride_set_num_threads(n)   Create a pool of n-1 persistent worker threads.
 *                               Thread 0 is the caller. Workers are pinned to
 *                               cores 1..n-1 (P-cores on Intel hybrid CPUs).
 *                               Call once before executing any plans.
 *                               n=1 or n=0: single-threaded (default).
 *
 *   stride_get_num_threads()    Query current thread count.
 *
 *   stride_pin_thread(core_id)  Pin the calling thread to a specific core.
 *                               Use core 0 for the caller when multithreading.
 *                               On i9-12th/13th/14th gen: cores 0-7 are P-cores.
 *
 * How it works
 * ------------
 * VectorFFT automatically selects a parallelization strategy per execute call:
 *
 *   K-SPLIT (K/T >= 256):
 *     Each thread processes a contiguous slice of the K batch dimension.
 *     Same plan, shared read-only twiddle tables, zero barriers, zero copies.
 *     Best for large K — each thread gets full work per codelet call.
 *
 *   GROUP-PARALLEL (K/T < 256):
 *     Each thread processes a subset of butterfly groups with full K lanes.
 *     Requires a spin-barrier between stages (groups in stage s+1 depend on s).
 *     Better for small K where K-split would cause cache line false sharing.
 *
 * The selection is automatic — the user just sets thread count and calls execute.
 *
 * Thread pool
 * -----------
 * Workers use spin-wait (not OS events) for lowest dispatch latency (~10ns).
 * This means idle workers burn CPU. For production use with intermittent FFT
 * calls, consider calling stride_set_num_threads(1) between bursts to release
 * the workers, then stride_set_num_threads(n) again before the next burst.
 *
 * Build
 * -----
 *   cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   cmake --build . --config Release
 *   ./bin/multithreaded_fft
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../src/stride-fft/core/env.h"
#include "../src/stride-fft/core/planner.h"

int main(void)
{
    unsigned int saved_mxcsr = stride_env_init();

    /* ── 1. Set up threading ── */
    int num_threads = 8;  /* Use 8 P-cores on i9-14900KF */
    stride_set_num_threads(num_threads);
    stride_pin_thread(0);  /* Pin caller to P-core 0 */
    printf("Threads: %d (workers pinned to cores 1..%d)\n\n",
           num_threads, num_threads - 1);

    /* ── 2. Create plan (same API as single-threaded) ── */
    stride_registry_t reg;
    stride_registry_init(&reg);

    const int N = 1000;
    const size_t K = 2048;  /* Large K for good K-split scaling */
    const size_t total = (size_t)N * K;

    double *re = (double *)stride_alloc(total * sizeof(double));
    double *im = (double *)stride_alloc(total * sizeof(double));

    /* Fill with test signal */
    for (int n = 0; n < N; n++) {
        double angle = -2.0 * M_PI * 3.0 * n / N;
        for (size_t k = 0; k < K; k++) {
            re[n * K + k] = cos(angle);
            im[n * K + k] = sin(angle);
        }
    }

    stride_plan_t *plan = stride_auto_plan(N, K, &reg);
    if (!plan) {
        fprintf(stderr, "Failed to create plan\n");
        return 1;
    }

    printf("Plan: N=%d K=%zu factors=", N, K);
    for (int i = 0; i < plan->num_stages; i++)
        printf("%s%d", i ? "x" : "", plan->factors[i]);
    printf("\n");

    /* ── 3. Execute (threading is automatic) ── */
    /* K/T = 2048/8 = 256 >= 256, so K-split strategy is used.
     * Each thread processes 256 contiguous lanes independently. */
    stride_execute_fwd(plan, re, im);

    /* Verify: bin 3 should have magnitude ~K */
    double mag3 = sqrt(re[3 * K] * re[3 * K] + im[3 * K] * im[3 * K]);
    printf("Forward: |X[3]| = %.1f (expect %.1f)\n", mag3, (double)K);

    /* ── 4. Backward + normalize ── */
    stride_execute_bwd(plan, re, im);
    for (size_t i = 0; i < total; i++) {
        re[i] /= N;
        im[i] /= N;
    }

    /* Check roundtrip */
    double max_err = 0.0;
    for (int n = 0; n < N; n++) {
        double angle = -2.0 * M_PI * 3.0 * n / N;
        double err_re = fabs(re[n * K] - cos(angle));
        double err_im = fabs(im[n * K] - sin(angle));
        if (err_re > max_err) max_err = err_re;
        if (err_im > max_err) max_err = err_im;
    }
    printf("Roundtrip max error: %.2e\n\n", max_err);

    /* ── 5. Benchmark: single vs multithreaded ── */
    printf("Benchmarking N=%d K=%zu...\n", N, K);

    /* Multithreaded (current setting) */
    int reps = 500;
    double t0 = 0, t1 = 0;

    /* Use a simple timing approach */
    {
        /* Warmup */
        for (int i = 0; i < 50; i++)
            stride_execute_fwd(plan, re, im);

        double start = (double)clock() / CLOCKS_PER_SEC;
        for (int i = 0; i < reps; i++)
            stride_execute_fwd(plan, re, im);
        double end = (double)clock() / CLOCKS_PER_SEC;
        t1 = (end - start) / reps * 1e9;
        printf("  %d threads: %.0f ns/iter\n", num_threads, t1);
    }

    /* Single-threaded */
    stride_set_num_threads(1);
    {
        for (int i = 0; i < 50; i++)
            stride_execute_fwd(plan, re, im);

        double start = (double)clock() / CLOCKS_PER_SEC;
        for (int i = 0; i < reps; i++)
            stride_execute_fwd(plan, re, im);
        double end = (double)clock() / CLOCKS_PER_SEC;
        t0 = (end - start) / reps * 1e9;
        printf("  1 thread:   %.0f ns/iter\n", t0);
    }

    if (t1 > 0)
        printf("  Speedup:    %.2fx\n", t0 / t1);

    /* ── 6. Small K example (group-parallel) ── */
    printf("\nSmall K example (group-parallel strategy):\n");
    stride_set_num_threads(num_threads);
    stride_plan_t *plan2 = stride_auto_plan(N, 256, &reg);
    if (plan2) {
        double *re2 = (double *)stride_alloc((size_t)N * 256 * sizeof(double));
        double *im2 = (double *)stride_alloc((size_t)N * 256 * sizeof(double));
        for (size_t i = 0; i < (size_t)N * 256; i++) {
            re2[i] = (double)rand() / RAND_MAX;
            im2[i] = 0;
        }

        /* K=256 T=8: K/T=32 < 256, so group-parallel is used.
         * Each thread processes all 256 lanes for a subset of groups,
         * with spin-barriers between stages. */
        stride_execute_fwd(plan2, re2, im2);
        printf("  N=%d K=256 T=%d: group-parallel (K/T=%d < 256)\n",
               N, num_threads, 256 / num_threads);

        stride_plan_destroy(plan2);
        stride_free(re2);
        stride_free(im2);
    }

    /* ── 7. Cleanup ── */
    stride_plan_destroy(plan);
    stride_free(re);
    stride_free(im);
    stride_set_num_threads(1);  /* release workers */
    stride_env_restore(saved_mxcsr);

    printf("\nDone.\n");
    return 0;
}
