/**
 * bench_vtune.c -- Single (N,K) profiling harness for Intel VTune
 *
 * Runs a single FFT size in a tight loop for a fixed duration.
 * Use with VTune's "Microarchitecture Exploration" or "Memory Access"
 * analysis to understand WHERE the CPU spends time.
 *
 * Usage:
 *   vfft_bench_vtune <N> <K> [library] [seconds] [--threads T] [--huge]
 *
 *   library: "ours" (default), "fftw", "mkl"
 *   seconds: how long to run (default: 5)
 *   --threads T: use T threads for VectorFFT (default: 1)
 *   --huge: use 2MB huge pages
 *
 * Examples:
 *   vtune -collect uarch-exploration -- vfft_bench_vtune 1000 256 ours 10
 *   vtune -collect memory-access    -- vfft_bench_vtune 4096 256 fftw 10
 *   vtune -collect uarch-exploration -- vfft_bench_vtune 256 32 mkl 10
 *
 * Compare results:
 *   Run for ours, fftw, mkl separately, then compare in VTune GUI.
 *   Look at: Retiring%, Front-End Bound%, Back-End Memory Bound%,
 *            L1/L2 miss rates, port utilization.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>

#ifdef VFFT_HAS_MKL
#include <mkl_dfti.h>
#endif

#include "../core/planner.h"
#include "../core/env.h"
#include "../core/compat.h"
#include "../core/threads.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <N> <K> [ours|fftw|mkl] [seconds]\n", argv[0]);
        printf("\nRun with VTune:\n");
        printf("  vtune -collect uarch-exploration -- %s 1000 256 ours 10\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    size_t K = (size_t)atoi(argv[2]);
    const char *lib = (argc > 3) ? argv[3] : "ours";
    double run_secs = (argc > 4) ? atof(argv[4]) : 5.0;
    int use_huge = 0;
    int num_threads = 1;
    /* Check for flags anywhere in args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--huge") == 0) use_huge = 1;
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            num_threads = atoi(argv[++i]);
    }

    size_t total = (size_t)N * K;
    size_t buf_bytes = total * sizeof(double);
    printf("N=%d  K=%zu  total=%zu  library=%s  duration=%.0fs  huge_pages=%s  threads=%d\n",
           N, K, total, lib, run_secs, use_huge ? "ON" : "off", num_threads);

    /* Initialize CPU environment (FTZ/DAZ) */
    stride_env_init();

    /* Allocate data buffers — huge pages eliminate DTLB overhead */
    double *re, *im;
    if (use_huge) {
        re = (double *)stride_alloc_huge(buf_bytes);
        im = (double *)stride_alloc_huge(buf_bytes);
        if (re && im)
            printf("Huge pages: allocated 2x %.1f MB\n", buf_bytes / (1024.0 * 1024.0));
        else
            printf("Huge pages: FAILED (falling back to standard)\n");
    } else {
        re = (double *)STRIDE_ALIGNED_ALLOC(64, buf_bytes);
        im = (double *)STRIDE_ALIGNED_ALLOC(64, buf_bytes);
    }
    if (!re || !im) {
        printf("ERROR: allocation failed\n");
        return 1;
    }

    srand(42);
    for (size_t i = 0; i < total; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* ================================================================
     * VectorFFT
     * ================================================================ */
    if (strcmp(lib, "ours") == 0) {
        stride_set_num_threads(num_threads);
        if (num_threads > 1) stride_pin_thread(0);

        stride_registry_t reg;
        stride_registry_init(&reg);

        stride_plan_t *plan = stride_auto_plan(N, K, &reg);
        if (!plan) {
            printf("ERROR: cannot create plan for N=%d K=%zu\n", N, K);
            return 1;
        }

        /* Print factorization */
        printf("Factorization: ");
        for (int s = 0; s < plan->num_stages; s++)
            printf("%s%d", s ? "x" : "", plan->factors[s]);
        printf("\n");

        /* Warmup */
        for (int i = 0; i < 20; i++)
            stride_execute_fwd(plan, re, im);

        /* Profiling loop */
        printf("Running VectorFFT for %.0f seconds...\n", run_secs);
        fflush(stdout);
        double t0 = now_ns();
        long long iters = 0;
        while ((now_ns() - t0) < run_secs * 1e9) {
            stride_execute_fwd(plan, re, im);
            iters++;
        }
        double elapsed = (now_ns() - t0) / 1e9;
        printf("Done: %lld iterations in %.1fs (%.1f ns/iter)\n",
               iters, elapsed, elapsed * 1e9 / iters);

        stride_plan_destroy(plan);
    }

    /* ================================================================
     * FFTW
     * ================================================================ */
    else if (strcmp(lib, "fftw") == 0) {
        fftw_iodim dim = {N, (int)K, (int)K};
        fftw_iodim batch = {(int)K, 1, 1};
        fftw_plan p = fftw_plan_guru_split_dft(1, &dim, 1, &batch,
                                                re, im, re, im, FFTW_MEASURE);
        if (!p) {
            printf("ERROR: FFTW plan failed\n");
            return 1;
        }

        /* Warmup */
        for (int i = 0; i < 20; i++)
            fftw_execute(p);

        printf("Running FFTW for %.0f seconds...\n", run_secs);
        fflush(stdout);
        double t0 = now_ns();
        long long iters = 0;
        while ((now_ns() - t0) < run_secs * 1e9) {
            fftw_execute_split_dft(p, re, im, re, im);
            iters++;
        }
        double elapsed = (now_ns() - t0) / 1e9;
        printf("Done: %lld iterations in %.1fs (%.1f ns/iter)\n",
               iters, elapsed, elapsed * 1e9 / iters);

        fftw_destroy_plan(p);
    }

    /* ================================================================
     * Intel MKL
     * ================================================================ */
#ifdef VFFT_HAS_MKL
    else if (strcmp(lib, "mkl") == 0) {
        DFTI_DESCRIPTOR_HANDLE h = NULL;
        DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
        DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(h, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
        MKL_LONG strides[2] = {0, (MKL_LONG)K};
        DftiSetValue(h, DFTI_INPUT_STRIDES, strides);
        DftiSetValue(h, DFTI_OUTPUT_STRIDES, strides);
        DftiSetValue(h, DFTI_INPUT_DISTANCE, 1);
        DftiSetValue(h, DFTI_OUTPUT_DISTANCE, 1);
        DftiSetValue(h, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
        DftiCommitDescriptor(h);

        double *ore = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));
        double *oim = (double *)STRIDE_ALIGNED_ALLOC(64, total * sizeof(double));

        /* Warmup */
        for (int i = 0; i < 20; i++)
            DftiComputeForward(h, re, im, ore, oim);

        printf("Running MKL for %.0f seconds...\n", run_secs);
        fflush(stdout);
        double t0 = now_ns();
        long long iters = 0;
        while ((now_ns() - t0) < run_secs * 1e9) {
            DftiComputeForward(h, re, im, ore, oim);
            iters++;
        }
        double elapsed = (now_ns() - t0) / 1e9;
        printf("Done: %lld iterations in %.1fs (%.1f ns/iter)\n",
               iters, elapsed, elapsed * 1e9 / iters);

        DftiFreeDescriptor(&h);
        STRIDE_ALIGNED_FREE(ore);
        STRIDE_ALIGNED_FREE(oim);
    }
#endif
    else {
        printf("Unknown library: %s (use ours, fftw, or mkl)\n", lib);
        return 1;
    }

    if (use_huge) {
        stride_free_huge(re, buf_bytes);
        stride_free_huge(im, buf_bytes);
    } else {
        STRIDE_ALIGNED_FREE(re);
        STRIDE_ALIGNED_FREE(im);
    }
    return 0;
}
