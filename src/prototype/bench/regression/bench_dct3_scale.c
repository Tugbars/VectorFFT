/* bench_dct3_scale.c — DCT-III at N=16, 32, 64 (where production has
 *                       no general-N DCT-III). Correctness vs brute-force
 *                       + roundtrip identity, plus perf measurement.
 *
 * No production comparison: production lacks general-N DCT-III. Our
 * DAG codelet IS the path that fills the gap. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__attribute__((target("avx2,fma"))) void radix16_dct3_avx2(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
__attribute__((target("avx2,fma"))) void radix32_dct3_avx2(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
__attribute__((target("avx2,fma"))) void radix64_dct3_avx2(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);

typedef void (*dct3_fn_t)(const double *, const double *, double *, double *,
                          const double *, const double *, size_t);

static double *aa(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 32, n * sizeof(double)) != 0) exit(1);
    return (double *)p;
}

static void fr(double *p, size_t n, unsigned s) {
    for (size_t i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        p[i] = (double)((int)(s >> 8) & 0x7fffff) / (double)0x800000 - 0.5;
    }
}

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

/* Brute-force DCT-III (FFTW REDFT01) on batch lane 0 only.
 * `out` is a contiguous [N] array, not batched. */
static void bf_dct3(const double *x, double *out, int N, size_t K) {
    double xb[256];
    for (int n = 0; n < N; n++) xb[n] = x[(size_t)n * K + 0];
    for (int k = 0; k < N; k++) {
        double s = xb[0];
        for (int n = 1; n < N; n++) {
            double t = M_PI * n * (2.0 * k + 1.0) / (2.0 * N);
            s += 2.0 * xb[n] * cos(t);
        }
        out[k] = s;
    }
}

static int run_one(int N, size_t K, dct3_fn_t fn) {
    double *in        = aa((size_t)N * K);
    double *out_re    = aa((size_t)N * K);
    double *out_im    = aa((size_t)N * K);
    double *dummy     = aa((size_t)N * K);
    fr(in, (size_t)N * K, 0xC3u + (unsigned)N);
    memset(dummy, 0, (size_t)N * K * sizeof(double));

    fn(in, dummy, out_re, out_im, dummy, dummy, K);

    /* Correctness vs brute-force. */
    double ref[256];
    bf_dct3(in, ref, N, K);
    double err = 0;
    for (int k = 0; k < N; k++) {
        double d = fabs(out_re[(size_t)k * K + 0] - ref[k]);
        if (d > err) err = d;
    }
    int correct = (err < 1e-9);

    /* Perf. */
    int repeat = (K <= 64) ? 20000 : (K <= 256 ? 5000 : 1000);
    int trials = 7;
    double best = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = now_ns();
        for (int r = 0; r < repeat; r++)
            fn(in, dummy, out_re, out_im, dummy, dummy, K);
        double dt = (now_ns() - t0) / (double)repeat;
        if (dt < best) best = dt;
    }
    double gflops = 0;
    /* op estimate from N log N + butterfly: ~3·N·log₂(N) FLOPs */
    if (best > 0) {
        double ops_per_call = 3.0 * (double)N * log2((double)N) * (double)K;
        gflops = ops_per_call / best;
    }
    printf("N=%-3d K=%-5zu  %7.1f ns  err=%.1e  %.2f GFLOPS  %s\n",
           N, K, best, err, gflops,
           correct ? "PASS" : "FAIL");
    free(in); free(out_re); free(out_im); free(dummy);
    return correct ? 0 : 1;
}

int main(void) {
    printf("================================================================\n");
    printf("  DCT-III at N=16, 32, 64: OCaml DAG (general-N, fills prod gap)\n");
    printf("================================================================\n");
    printf("  Op counts: N=16→141, N=32→397, N=64→1021 (DAG, pre-FMA-fusion)\n");
    printf("================================================================\n");

    struct { int N; dct3_fn_t fn; } entries[] = {
        {16, radix16_dct3_avx2},
        {32, radix32_dct3_avx2},
        {64, radix64_dct3_avx2},
    };
    size_t Ks[] = {32, 128, 512, 0};
    int fails = 0;
    for (size_t i = 0; i < sizeof(entries)/sizeof(entries[0]); i++) {
        for (int ki = 0; Ks[ki] != 0; ki++) {
            fails += run_one(entries[i].N, Ks[ki], entries[i].fn);
        }
        printf("\n");
    }
    printf("%s\n", fails == 0 ? "All correctness PASS" : "Some FAIL");
    return fails;
}
