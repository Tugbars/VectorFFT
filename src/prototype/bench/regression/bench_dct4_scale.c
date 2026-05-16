/* bench_dct4_scale.c — DCT-IV at N=8,16,32,64: correctness vs brute-force
 *                       + roundtrip identity DCT-IV(DCT-IV(x)) = 2N·x.
 *
 * Production has Lee 1984 implementation in src/core/dct4.h but no
 * specialized N=8 codelet. Our fused DAG codelet is the alternative to
 * production's runtime 3-pass (pre-twiddle + c2c-N/2 IFFT + post-twiddle). */

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

__attribute__((target("avx2,fma"))) void radix8_dct4_avx2(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
__attribute__((target("avx2,fma"))) void radix16_dct4_avx2(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
__attribute__((target("avx2,fma"))) void radix32_dct4_avx2(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
__attribute__((target("avx2,fma"))) void radix64_dct4_avx2(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);

typedef void (*dct4_fn_t)(const double *, const double *, double *, double *,
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

/* Brute-force DCT-IV (FFTW REDFT11): Y[k] = 2·Σ x[n]·cos(π(2k+1)(2n+1)/(4N))
 * batch lane 0 only, out is contiguous [N]. */
static void bf_dct4(const double *x, double *out, int N, size_t K) {
    double xb[256];
    for (int n = 0; n < N; n++) xb[n] = x[(size_t)n * K + 0];
    for (int k = 0; k < N; k++) {
        double s = 0;
        for (int n = 0; n < N; n++) {
            double t = M_PI * (2.0 * k + 1.0) * (2.0 * n + 1.0) / (4.0 * N);
            s += 2.0 * xb[n] * cos(t);
        }
        out[k] = s;
    }
}

static int run_one(int N, size_t K, dct4_fn_t fn) {
    double *in        = aa((size_t)N * K);
    double *out_re    = aa((size_t)N * K);
    double *out_re2   = aa((size_t)N * K);
    double *out_im    = aa((size_t)N * K);
    double *dummy     = aa((size_t)N * K);
    fr(in, (size_t)N * K, 0xC4u + (unsigned)N);
    memset(dummy, 0, (size_t)N * K * sizeof(double));

    /* Forward DCT-IV */
    fn(in, dummy, out_re, out_im, dummy, dummy, K);

    /* Correctness vs brute-force on batch lane 0. */
    double ref[256];
    bf_dct4(in, ref, N, K);
    double err = 0;
    for (int k = 0; k < N; k++) {
        double d = fabs(out_re[(size_t)k * K + 0] - ref[k]);
        if (d > err) err = d;
    }

    /* Roundtrip: DCT-IV(DCT-IV(x)) = 2N · x. */
    fn(out_re, dummy, out_re2, out_im, dummy, dummy, K);
    double rt_err = 0;
    double scale = 1.0 / (2.0 * (double)N);
    for (size_t i = 0; i < (size_t)N * K; i++) {
        double d = fabs(out_re2[i] * scale - in[i]);
        if (d > rt_err) rt_err = d;
    }

    int correct = (err < 1e-9) && (rt_err < 1e-9);

    /* Perf */
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
    if (best > 0) {
        double ops_per_call = 3.0 * (double)N * log2((double)N) * (double)K;
        gflops = ops_per_call / best;
    }
    printf("N=%-3d K=%-5zu  %7.1f ns  bf_err=%.1e  rt_err=%.1e  %.2f GFLOPS  %s\n",
           N, K, best, err, rt_err, gflops, correct ? "PASS" : "FAIL");
    free(in); free(out_re); free(out_re2); free(out_im); free(dummy);
    return correct ? 0 : 1;
}

int main(void) {
    printf("================================================================\n");
    printf("  DCT-IV at N=8,16,32,64: OCaml DAG (Lee 1984, fused codelet)\n");
    printf("================================================================\n");
    printf("  Op counts: N=8→57, N=16→145, N=32→352, N=64→835 (pre-FMA-fuse)\n");
    printf("  bf_err = vs brute-force; rt_err = vs roundtrip (DCT-IV² / 2N)\n");
    printf("================================================================\n");

    struct { int N; dct4_fn_t fn; } entries[] = {
        {8,  radix8_dct4_avx2},
        {16, radix16_dct4_avx2},
        {32, radix32_dct4_avx2},
        {64, radix64_dct4_avx2},
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
