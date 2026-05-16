/* bench_r32.c — R=32 t1_dit Hand vs Topo, K-sweep, single process. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

#include "../radix32_handcoded.h"

__attribute__((target("avx512f")))
void radix32_t1_dit_fwd_avx512(
    double * __restrict__, double * __restrict__,
    const double * __restrict__, const double * __restrict__,
    size_t, size_t);

static double *aa(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc fail\n"); exit(1);
    }
    return (double *)p;
}
static void fr(double *p, size_t n, unsigned s) {
    for (size_t i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        p[i] = (double)((int)(s >> 8) & 0x7fffff) / (double)0x800000 - 0.5;
    }
}
static double max_rel(const double *a, const double *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        double s = fabs(a[i]) + fabs(b[i]) + 1e-30;
        double r = d / s;
        if (r > m) m = r;
    }
    return m;
}
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}
static double bn(void (*f)(void), int repeat, int trials) {
    double best = 1e18;
    for (int i = 0; i < 100; i++) f();
    for (int t = 0; t < trials; t++) {
        double t0 = now();
        for (int i = 0; i < repeat; i++) f();
        double dt = (now() - t0) / (double)repeat;
        if (dt < best) best = dt;
    }
    return best;
}

static size_t g_K;
static double *g_rio_re_h, *g_rio_im_h;
static double *g_rio_re_t, *g_rio_im_t;
static double *g_tw_re, *g_tw_im;
static double *g_in_re, *g_in_im;

static void cH(void) { radix32_t1_dit_fwd_avx512(g_rio_re_h, g_rio_im_h, g_tw_re, g_tw_im, g_K, g_K); }
static void cT(void) { radix32_t1_dit_fwd_avx512(g_rio_re_t, g_rio_im_t, g_tw_re, g_tw_im, g_K, g_K); }

int main(int argc, char **argv) {
    g_K = (argc > 1) ? (size_t)atoi(argv[1]) : 1024;
    if (g_K < 8 || g_K % 8 != 0) { fprintf(stderr, "K mod 8\n"); return 1; }

    g_in_re = aa(32 * g_K);
    g_in_im = aa(32 * g_K);
    g_rio_re_h = aa(32 * g_K); g_rio_im_h = aa(32 * g_K);
    g_rio_re_t = aa(32 * g_K); g_rio_im_t = aa(32 * g_K);
    g_tw_re = aa(31 * g_K); g_tw_im = aa(31 * g_K);

    fr(g_in_re, 32 * g_K, 0xa1);
    fr(g_in_im, 32 * g_K, 0xa2);
    fr(g_tw_re, 31 * g_K, 0xb1);
    fr(g_tw_im, 31 * g_K, 0xb2);

    memcpy(g_rio_re_h, g_in_re, 32 * g_K * sizeof(double));
    memcpy(g_rio_im_h, g_in_im, 32 * g_K * sizeof(double));
    memcpy(g_rio_re_t, g_in_re, 32 * g_K * sizeof(double));
    memcpy(g_rio_im_t, g_in_im, 32 * g_K * sizeof(double));

    cH(); cT();

    double e = max_rel(g_rio_re_h, g_rio_re_t, 32 * g_K);
    double e2 = max_rel(g_rio_im_h, g_rio_im_t, 32 * g_K);
    double err = e > e2 ? e : e2;
    if (err > 1e-8) {
        printf("CORRECTNESS FAIL: re=%.2e im=%.2e\n", e, e2);
        return 2;
    }

    int repeat = 3000, trials = 7;
    double tH = bn(cH, repeat, trials);
    double tT = bn(cT, repeat, trials);
    printf("K=%5zu  Hand=%8.1f  Topo=%8.1f  | T/H=%.3f\n", g_K, tH, tT, tT/tH);
    return 0;
}
