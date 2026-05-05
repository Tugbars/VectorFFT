/* bench_compiler.c — single-cell timing harness for compiler comparison.
 *
 * Same source compiled by MSVC / ICX / GCC and linked against each
 * compiler's vfft library (build_{msvc,icx,gcc}/lib/...). Times one
 * representative cell — N=1024 K=256 — with VFFT_MEASURE wisdom.
 *
 * Min over 21 reps after 5 warmups, single-threaded P-core pinned.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"

#define N    65536
#define K    4
#define WARM 5
#define REPS 21

static double now_ns(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}

int main(int argc, char **argv) {
    const char *tag = (argc > 1) ? argv[1] : "(unknown)";

    vfft_init();
    vfft_pin_thread(0);
    vfft_set_num_threads(1);

    /* Wisdom is in build_tuned/. Try a few likely paths. */
    int rc = vfft_load_wisdom("build_tuned/vfft_wisdom_tuned.txt");
    if (rc != 0) rc = vfft_load_wisdom("../build_tuned/vfft_wisdom_tuned.txt");
    if (rc != 0) rc = vfft_load_wisdom(
        "C:/Users/Tugbars/Desktop/highSpeedFFT/build_tuned/vfft_wisdom_tuned.txt");
    if (rc != 0) {
        fprintf(stderr, "[%s] no wisdom — falling back to ESTIMATE\n", tag);
    }

    vfft_plan p = vfft_plan_c2c(N, K, rc == 0 ? VFFT_MEASURE : VFFT_ESTIMATE);
    if (!p) { fprintf(stderr, "[%s] plan failed\n", tag); return 1; }

    size_t NK = (size_t)N * K;
    double *re = (double *)vfft_alloc(NK * sizeof(double));
    double *im = (double *)vfft_alloc(NK * sizeof(double));
    if (!re || !im) { fprintf(stderr, "[%s] alloc failed\n", tag); return 1; }

    srand(42);
    for (size_t i = 0; i < NK; i++) {
        re[i] = (double)rand() / RAND_MAX - 0.5;
        im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    /* Warmup */
    for (int w = 0; w < WARM; w++) vfft_execute_fwd(p, re, im);

    /* Min over REPS */
    double best = 1e30;
    for (int r = 0; r < REPS; r++) {
        double t0 = now_ns();
        vfft_execute_fwd(p, re, im);
        double dt = now_ns() - t0;
        if (dt < best) best = dt;
    }

    /* GFLOP/s estimate: 5 N log2(N) K flops per forward FFT */
    double gf = 5.0 * N * log2((double)N) * K / best;

    printf("[%s]  N=%d K=%d  min=%9.0f ns  %6.2f GFLOP/s  (wisdom=%s)\n",
           tag, N, K, best, gf, rc == 0 ? "yes" : "no");

    vfft_free(re); vfft_free(im); vfft_destroy(p);
    return 0;
}
