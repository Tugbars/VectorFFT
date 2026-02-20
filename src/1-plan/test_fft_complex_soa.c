/**
 * @file test_fft_complex_soa.c
 * @brief Correctness tests + benchmark for AoS ↔ SoA conversion
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "fft_complex_soa.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); g_fail++; return; } \
} while(0)

#define TOL 1e-15

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/*── correctness: roundtrip AoS → SoA → AoS ──*/
static void test_roundtrip(int n) {
    double *aos  = (double*)aligned_alloc(64, 2*n*sizeof(double));
    double *re   = (double*)aligned_alloc(64, n*sizeof(double));
    double *im   = (double*)aligned_alloc(64, n*sizeof(double));
    double *aos2 = (double*)aligned_alloc(64, 2*n*sizeof(double));

    srand(42 + n);
    for (int i = 0; i < 2*n; i++)
        aos[i] = (double)rand()/RAND_MAX - 0.5;

    fft_deinterleave(aos, re, im, n);

    /* Verify split correctness */
    for (int i = 0; i < n; i++) {
        if (fabs(re[i] - aos[2*i]) > TOL || fabs(im[i] - aos[2*i+1]) > TOL) {
            printf("  FAIL: deinterleave mismatch at i=%d (n=%d)\n", i, n);
            g_fail++; goto cleanup;
        }
    }

    fft_reinterleave(re, im, aos2, n);

    /* Verify roundtrip */
    for (int i = 0; i < 2*n; i++) {
        if (fabs(aos[i] - aos2[i]) > TOL) {
            printf("  FAIL: roundtrip mismatch at i=%d (n=%d)\n", i, n);
            g_fail++; goto cleanup;
        }
    }
    g_pass++;

cleanup:
    free(aos); free(re); free(im); free(aos2);
}

/*── benchmark ──*/
static void bench(int n) {
    double *aos = (double*)aligned_alloc(64, 2*n*sizeof(double));
    double *re  = (double*)aligned_alloc(64, n*sizeof(double));
    double *im  = (double*)aligned_alloc(64, n*sizeof(double));
    double *aos2= (double*)aligned_alloc(64, 2*n*sizeof(double));

    srand(42);
    for (int i = 0; i < 2*n; i++) aos[i] = (double)rand()/RAND_MAX;

    /* Warmup */
    double t0 = now_ns();
    while (now_ns() - t0 < 100e6) {
        fft_deinterleave(aos, re, im, n);
        fft_reinterleave(re, im, aos2, n);
    }

    /* Deinterleave bench */
    t0 = now_ns(); long it = 0;
    while (now_ns() - t0 < 300e6) { fft_deinterleave(aos, re, im, n); it++; }
    double deint_ns = (now_ns() - t0) / it;

    /* Reinterleave bench */
    t0 = now_ns(); it = 0;
    while (now_ns() - t0 < 300e6) { fft_reinterleave(re, im, aos2, n); it++; }
    double reint_ns = (now_ns() - t0) / it;

    double total_bytes = n * 2.0 * sizeof(double); /* read + write */
    double deint_gbps = (total_bytes * 2.0) / deint_ns;  /* read interleaved + write split */
    double reint_gbps = (total_bytes * 2.0) / reint_ns;

    printf("  N=%-8d deint: %8.1f ns (%5.1f GB/s)  reint: %8.1f ns (%5.1f GB/s)  total: %8.1f ns\n",
           n, deint_ns, deint_gbps, reint_ns, reint_gbps, deint_ns + reint_ns);

    free(aos); free(re); free(im); free(aos2);
}

int main(void) {
    printf("============================================\n");
    printf(" AoS ↔ SoA Conversion Tests\n");
    printf(" SIMD: %s\n", fft_soa_get_simd_capabilities());
    printf("============================================\n\n");

    /* Correctness: sweep sizes including all remainder cases */
    printf("[Correctness — roundtrip]\n");
    int sizes[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 31, 32, 33,
                   63, 64, 65, 127, 128, 255, 256, 512, 1000, 1024, 4096,
                   4097, 8191, 8192, 65536};
    int nsizes = sizeof(sizes)/sizeof(sizes[0]);

    for (int i = 0; i < nsizes; i++)
        test_roundtrip(sizes[i]);

    printf("  %d/%d roundtrip tests passed\n\n", g_pass, g_pass + g_fail);

    /* Benchmark */
    printf("[Benchmark]\n");
    int bsizes[] = {16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576};
    int nbsizes = sizeof(bsizes)/sizeof(bsizes[0]);
    for (int i = 0; i < nbsizes; i++)
        bench(bsizes[i]);

    printf("\n============================================\n");
    printf(" %d passed, %d failed\n", g_pass, g_fail);
    printf("============================================\n");

    return g_fail ? 1 : 0;
}
