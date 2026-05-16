/* bench_trig_scale.c — DCT-II / DST-II / DST-III at N=16,32,64.
 *
 * Production has specialized N=8 codelets only; at N>8 production runtime
 * is a 3-pass (sign-flip + Makhoul DCT + reverse) on top of rdft+butterfly,
 * i.e. ~5 memory passes for DST. Our fused DAG collapses everything to a
 * single straight-line codelet.
 *
 * This bench validates correctness against brute force and reports best-ns
 * + GFLOPS at each (N, K) — confirming the N=8 win pattern scales. */

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

typedef void (*fn_t)(const double *, const double *, double *, double *,
                     const double *, const double *, size_t);

#define DECL(name) __attribute__((target("avx2,fma"))) void name(\
    const double *, const double *, double *, double *,\
    const double *, const double *, size_t)

DECL(radix16_dct2_avx2); DECL(radix32_dct2_avx2); DECL(radix64_dct2_avx2);
DECL(radix16_dst2_avx2); DECL(radix32_dst2_avx2); DECL(radix64_dst2_avx2);
DECL(radix16_dst3_avx2); DECL(radix32_dst3_avx2); DECL(radix64_dst3_avx2);

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

/* Brute-force lane-0 references. */
static void bf_dct2(const double *x, double *out, int N, size_t K) {
    double xb[256];
    for (int n = 0; n < N; n++) xb[n] = x[(size_t)n * K + 0];
    for (int k = 0; k < N; k++) {
        double s = 0;
        for (int n = 0; n < N; n++) {
            double t = M_PI * k * (2.0 * n + 1.0) / (2.0 * N);
            s += 2.0 * xb[n] * cos(t);
        }
        out[k] = s;
    }
}
static void bf_dst2(const double *x, double *out, int N, size_t K) {
    double xb[256];
    for (int n = 0; n < N; n++) xb[n] = x[(size_t)n * K + 0];
    for (int k = 0; k < N; k++) {
        double s = 0;
        for (int n = 0; n < N; n++) {
            double t = M_PI * (k + 1.0) * (2.0 * n + 1.0) / (2.0 * N);
            s += 2.0 * xb[n] * sin(t);
        }
        out[k] = s;
    }
}
static void bf_dst3(const double *x, double *out, int N, size_t K) {
    double xb[256];
    for (int n = 0; n < N; n++) xb[n] = x[(size_t)n * K + 0];
    for (int k = 0; k < N; k++) {
        double s = ((k & 1) ? -1.0 : 1.0) * xb[N - 1];
        for (int n = 0; n < N - 1; n++) {
            double t = M_PI * (n + 1.0) * (2.0 * k + 1.0) / (2.0 * N);
            s += 2.0 * xb[n] * sin(t);
        }
        out[k] = s;
    }
}

typedef void (*bf_fn_t)(const double *, double *, int, size_t);

static int run_one(const char *name, int N, size_t K, fn_t fn, bf_fn_t bf) {
    double *in     = aa((size_t)N * K);
    double *out_re = aa((size_t)N * K);
    double *out_im = aa((size_t)N * K);
    double *dummy  = aa((size_t)N * K);
    fr(in, (size_t)N * K, 0xB0u + (unsigned)N + (unsigned)(name[5]));
    memset(dummy, 0, (size_t)N * K * sizeof(double));

    fn(in, dummy, out_re, out_im, dummy, dummy, K);

    double ref[256];
    bf(in, ref, N, K);
    double err = 0;
    for (int k = 0; k < N; k++) {
        double d = fabs(out_re[(size_t)k * K + 0] - ref[k]);
        if (d > err) err = d;
    }
    int correct = (err < 1e-9);

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
    double ops_per_call = 3.0 * (double)N * log2((double)N) * (double)K;
    double gflops = (best > 0) ? ops_per_call / best : 0.0;
    printf("%-7s N=%-3d K=%-5zu  %8.1f ns  err=%.1e  %5.2f GFLOPS  %s\n",
           name, N, K, best, err, gflops, correct ? "PASS" : "FAIL");
    free(in); free(out_re); free(out_im); free(dummy);
    return correct ? 0 : 1;
}

int main(void) {
    printf("================================================================\n");
    printf("  DCT-II / DST-II / DST-III at N=16,32,64 — OCaml fused DAG\n");
    printf("  (production has no specialized codelet at these sizes)\n");
    printf("================================================================\n");
    struct row { const char *name; int N; fn_t fn; bf_fn_t bf; } rows[] = {
        {"DCT-II",  16, radix16_dct2_avx2, bf_dct2},
        {"DCT-II",  32, radix32_dct2_avx2, bf_dct2},
        {"DCT-II",  64, radix64_dct2_avx2, bf_dct2},
        {"DST-II",  16, radix16_dst2_avx2, bf_dst2},
        {"DST-II",  32, radix32_dst2_avx2, bf_dst2},
        {"DST-II",  64, radix64_dst2_avx2, bf_dst2},
        {"DST-III", 16, radix16_dst3_avx2, bf_dst3},
        {"DST-III", 32, radix32_dst3_avx2, bf_dst3},
        {"DST-III", 64, radix64_dst3_avx2, bf_dst3},
    };
    size_t Ks[] = {32, 128, 512, 0};
    int fails = 0;
    const char *last = "";
    for (size_t i = 0; i < sizeof(rows)/sizeof(rows[0]); i++) {
        if (strcmp(rows[i].name, last) != 0) {
            if (i != 0) printf("\n");
            last = rows[i].name;
        }
        for (int ki = 0; Ks[ki] != 0; ki++) {
            fails += run_one(rows[i].name, rows[i].N, Ks[ki], rows[i].fn, rows[i].bf);
        }
    }
    printf("\n%s\n", fails == 0 ? "All correctness PASS" : "Some FAIL");
    return fails;
}
