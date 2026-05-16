/* bench_r16_su.c — Hand, Topo, SU side-by-side for R=16 t1_dit on AVX-512. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

#include "../radix16_handcoded.h"

__attribute__((target("avx512f")))
void radix16_t1_dit_fwd_avx512(
    double * __restrict__, double * __restrict__,
    const double * __restrict__, const double * __restrict__,
    size_t, size_t);

__attribute__((target("avx512f")))
void radix16_t1_dit_fwd_avx512(
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
static double *g_rio_re_s, *g_rio_im_s;
static double *g_tw_re, *g_tw_im;
static double *g_in_re, *g_in_im;

static void cH(void) { radix16_t1_dit_fwd_avx512(g_rio_re_h, g_rio_im_h, g_tw_re, g_tw_im, g_K, g_K); }
static void cT(void) { radix16_t1_dit_fwd_avx512(g_rio_re_t, g_rio_im_t, g_tw_re, g_tw_im, g_K, g_K); }
static void cS(void) { radix16_t1_dit_fwd_avx512(g_rio_re_s, g_rio_im_s, g_tw_re, g_tw_im, g_K, g_K); }

int main(int argc, char **argv) {
    g_K = (argc > 1) ? (size_t)atoi(argv[1]) : 1024;
    if (g_K < 8 || g_K % 8 != 0) { fprintf(stderr, "K mod 8\n"); return 1; }

    g_in_re = aa(16 * g_K);
    g_in_im = aa(16 * g_K);
    g_rio_re_h = aa(16 * g_K); g_rio_im_h = aa(16 * g_K);
    g_rio_re_t = aa(16 * g_K); g_rio_im_t = aa(16 * g_K);
    g_rio_re_s = aa(16 * g_K); g_rio_im_s = aa(16 * g_K);
    g_tw_re = aa(15 * g_K); g_tw_im = aa(15 * g_K);

    fr(g_in_re, 16 * g_K, 0xa1);
    fr(g_in_im, 16 * g_K, 0xa2);
    fr(g_tw_re, 15 * g_K, 0xb1);
    fr(g_tw_im, 15 * g_K, 0xb2);

    memcpy(g_rio_re_h, g_in_re, 16 * g_K * sizeof(double));
    memcpy(g_rio_im_h, g_in_im, 16 * g_K * sizeof(double));
    memcpy(g_rio_re_t, g_in_re, 16 * g_K * sizeof(double));
    memcpy(g_rio_im_t, g_in_im, 16 * g_K * sizeof(double));
    memcpy(g_rio_re_s, g_in_re, 16 * g_K * sizeof(double));
    memcpy(g_rio_im_s, g_in_im, 16 * g_K * sizeof(double));

    cH(); cT(); cS();

    double e_t = max_rel(g_rio_re_h, g_rio_re_t, 16 * g_K);
    double e_s = max_rel(g_rio_re_h, g_rio_re_s, 16 * g_K);
    if (e_t > 1e-9 || e_s > 1e-9) {
        printf("CORRECTNESS FAIL: topo=%.2e su=%.2e\n", e_t, e_s);
        return 2;
    }

    int repeat = 5000, trials = 7;
    double tH = bn(cH, repeat, trials);
    double tT = bn(cT, repeat, trials);
    double tS = bn(cS, repeat, trials);
    printf("K=%5zu  Hand=%7.1f  Topo=%7.1f  SU=%7.1f  | T/H=%.3f  S/H=%.3f  S/T=%.3f\n",
           g_K, tH, tT, tS, tT/tH, tS/tH, tS/tT);
    return 0;
}
