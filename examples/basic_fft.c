/*
 * basic_fft.c -- VectorFFT complete API example
 *
 * Demonstrates every public API entry point:
 *
 *   Environment (env.h)
 *   -------------------
 *   stride_env_init()         Set FTZ+DAZ for consistent SIMD performance.
 *                             Returns previous MXCSR state for later restore.
 *   stride_env_restore(mxcsr) Restore original FPU state.
 *   stride_alloc(bytes)       64-byte aligned allocation (SIMD-friendly).
 *   stride_alloc_huge(bytes)  2MB huge-page allocation with fallback to
 *                             stride_alloc. Reduces TLB pressure for large N*K.
 *   stride_free(ptr)          Free memory from stride_alloc.
 *   stride_free_huge(ptr, sz) Free memory from stride_alloc_huge.
 *   stride_print_info()       Print library version, ISA, and CPU info.
 *   stride_set_verbose(lvl)   Set verbosity (0=silent, 1=info, 2=debug).
 *   stride_pin_thread(core)   Pin calling thread to a specific core.
 *   stride_unpin_thread()     Unpin calling thread.
 *   stride_get_num_cores()    Query physical core count.
 *
 *   Registry (registry.h)
 *   ---------------------
 *   stride_registry_init(reg) Populate the codelet registry. Selects the
 *                             best ISA at compile time (AVX-512 > AVX2 > scalar).
 *                             The registry maps each radix (2..64) to its
 *                             n1 (no-twiddle) and t1 (twiddle) codelet pair.
 *
 *   Planner (planner.h)
 *   -------------------
 *   stride_auto_plan(N,K,reg) Create an FFT plan using the heuristic factorizer.
 *                             Instant, no benchmarking. Good default for most uses.
 *                             Handles composite N, prime N (Rader/Bluestein), any K.
 *
 *   stride_wise_plan(N,K,reg,wis)
 *                             Create a plan using cached wisdom if available,
 *                             otherwise falls back to heuristic. Best of both worlds.
 *
 *   stride_exhaustive_plan(N,K,reg,factors,nf)
 *                             Create a plan from an explicit factorization.
 *                             Used internally by the calibrator.
 *
 *   stride_wisdom_init(wis)   Initialize an empty wisdom database.
 *   stride_wisdom_load(wis,path)
 *                             Load wisdom from file. Returns 0 on success.
 *   stride_wisdom_save(wis,path)
 *                             Save wisdom to file. Returns 0 on success.
 *   stride_wisdom_calibrate(wis,N,K,reg)
 *                             Benchmark all factorizations for (N,K) and store
 *                             the best in the wisdom database.
 *
 *   Executor (executor.h)
 *   ---------------------
 *   stride_execute_fwd(plan,re,im)
 *                             Execute forward (DIT) FFT in-place.
 *                             Input: N*K doubles in re[], N*K in im[].
 *                             Output: DFT coefficients in digit-reversed order.
 *
 *   stride_execute_bwd(plan,re,im)
 *                             Execute backward (DIF) FFT in-place.
 *                             Output is in natural order (permutation-free roundtrip).
 *                             Divide by N to normalize.
 *
 *   stride_plan_destroy(plan) Free all memory associated with a plan.
 *
 * Data layout
 * -----------
 * VectorFFT uses split-complex format: separate re[] and im[] arrays,
 * each of size N * K doubles. K is the batch size -- the number of
 * independent FFTs computed simultaneously. Data is stored as:
 *
 *   re[n * K + k] = real part of sample n, batch element k
 *   im[n * K + k] = imaginary part of sample n, batch element k
 *
 * All transforms are in-place: input and output share the same buffer.
 *
 * Build
 * -----
 *   cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release -DVFFT_BUILD_EXAMPLES=ON
 *   cmake --build . --config Release
 *   ./bin/basic_fft
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* VectorFFT headers (order matters) */
#include "../src/stride-fft/core/env.h"
#include "../src/stride-fft/core/planner.h"

int main(void)
{
    /* ── 1. Environment setup ── */
    unsigned int saved_mxcsr = stride_env_init();  /* FTZ+DAZ for SIMD */
    stride_set_verbose(1);
    stride_print_info();
    printf("\n");

    /* ── 2. Registry ── */
    stride_registry_t reg;
    stride_registry_init(&reg);

    /* ── 3. Allocate data (split-complex) ── */
    const int N = 1000;       /* FFT length */
    const size_t K = 256;     /* batch size */
    const size_t total = (size_t)N * K;

    double *re = (double *)stride_alloc(total * sizeof(double));
    double *im = (double *)stride_alloc(total * sizeof(double));

    /* Fill with a test signal: single tone at bin 7 */
    for (int n = 0; n < N; n++) {
        double angle = -2.0 * M_PI * 7.0 * n / N;
        for (size_t k = 0; k < K; k++) {
            re[n * K + k] = cos(angle);
            im[n * K + k] = sin(angle);
        }
    }

    /* ── 4. Plan (heuristic -- instant) ── */
    stride_plan_t *plan = stride_auto_plan(N, K, &reg);
    if (!plan) {
        fprintf(stderr, "Failed to create plan for N=%d K=%zu\n", N, K);
        return 1;
    }

    printf("Plan: N=%d K=%zu factors=", N, K);
    for (int i = 0; i < plan->num_stages; i++)
        printf("%s%d", i ? "x" : "", plan->factors[i]);
    printf("\n");

    /* ── 5. Forward FFT ── */
    stride_execute_fwd(plan, re, im);

    /* Check: bin 7 should have magnitude ~K, all others ~0 */
    double mag7 = sqrt(re[7 * K] * re[7 * K] + im[7 * K] * im[7 * K]);
    double mag0 = sqrt(re[0] * re[0] + im[0] * im[0]);
    printf("Forward: |X[7]| = %.6f (expect %.1f), |X[0]| = %.2e (expect ~0)\n",
           mag7, (double)K, mag0);

    /* ── 6. Backward FFT (inverse) ── */
    stride_execute_bwd(plan, re, im);

    /* Normalize by N */
    for (size_t i = 0; i < total; i++) {
        re[i] /= N;
        im[i] /= N;
    }

    /* Check roundtrip: should recover original signal */
    double max_err = 0.0;
    for (int n = 0; n < N; n++) {
        double angle = -2.0 * M_PI * 7.0 * n / N;
        double err_re = fabs(re[n * K] - cos(angle));
        double err_im = fabs(im[n * K] - sin(angle));
        if (err_re > max_err) max_err = err_re;
        if (err_im > max_err) max_err = err_im;
    }
    printf("Roundtrip max error: %.2e (expect ~1e-14)\n\n", max_err);

    stride_plan_destroy(plan);

    /* ── 7. Wisdom-based planning (optional) ── */
    stride_wisdom_t wis;
    stride_wisdom_init(&wis);

    /* Try loading existing wisdom */
    if (stride_wisdom_load(&wis, "vfft_wisdom.txt") == 0 && wis.count > 0) {
        printf("Loaded %d wisdom entries\n", wis.count);
    } else {
        printf("No wisdom file found -- calibrating N=%d K=%zu...\n", N, K);
        stride_wisdom_calibrate(&wis, N, K, &reg);
        stride_wisdom_save(&wis, "vfft_wisdom.txt");
        printf("Wisdom saved (%d entries)\n", wis.count);
    }

    /* Create plan from wisdom (falls back to heuristic if not found) */
    stride_plan_t *wplan = stride_wise_plan(N, K, &reg, &wis);
    printf("Wisdom plan: factors=");
    for (int i = 0; i < wplan->num_stages; i++)
        printf("%s%d", i ? "x" : "", wplan->factors[i]);
    printf("\n");

    stride_plan_destroy(wplan);

    /* ── 8. Prime-N support ── */
    /* Rader: prime where N-1 is smooth (e.g. 61, since 60 = 2^2 * 3 * 5) */
    stride_plan_t *rader_plan = stride_auto_plan(61, K, &reg);
    printf("\nPrime N=61 (Rader): factors=");
    for (int i = 0; i < rader_plan->num_stages; i++)
        printf("%s%d", i ? "x" : "", rader_plan->factors[i]);
    printf("\n");
    stride_plan_destroy(rader_plan);

    /* Bluestein: prime where N-1 is not smooth (e.g. 509) */
    stride_plan_t *blue_plan = stride_auto_plan(509, K, &reg);
    if (blue_plan) {
        printf("Prime N=509 (Bluestein): factors=");
        for (int i = 0; i < blue_plan->num_stages; i++)
            printf("%s%d", i ? "x" : "", blue_plan->factors[i]);
        printf("\n");
        stride_plan_destroy(blue_plan);
    }

    /* ── 9. Cleanup ── */
    stride_free(re);
    stride_free(im);
    stride_env_restore(saved_mxcsr);

    printf("\nDone.\n");
    return 0;
}
