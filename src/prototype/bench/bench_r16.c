/* bench_r16.c — R=16 t1_dit: Hand vs Topo, side-by-side timing.
 *
 *   Hand = radix16_t1_dit_fwd_avx512 from gen_radix16.py (with stack spills)
 *   Topo = radix16_t1_dit_fwd_avx512_gen_inplace (topological order, no spill modeling)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

/* User's hand-coded R=16 t1_dit codelet. */
#include "../radix16_handcoded.h"

__attribute__((target("avx512f")))
void radix16_t1_dit_fwd_avx512_gen_inplace(
    double * __restrict__, double * __restrict__,
    const double * __restrict__, const double * __restrict__,
    size_t, size_t);

static double *aligned_alloc_doubles(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "posix_memalign failed\n"); exit(1);
    }
    return (double *)p;
}

static void fill_random(double *p, size_t n, unsigned seed) {
    unsigned s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        p[i] = (double)((int)(s >> 8) & 0x7fffff) / (double)0x800000 - 0.5;
    }
}

static double max_rel_err(const double *a, const double *b, size_t n) {
    double max = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        double s = fabs(a[i]) + fabs(b[i]) + 1e-30;
        double r = d / s;
        if (r > max) max = r;
    }
    return max;
}

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static double bench(void (*fn)(void), int repeat, int trials) {
    double best = 1e18;
    for (int i = 0; i < 100; i++) fn();
    for (int t = 0; t < trials; t++) {
        double t0 = now_ns();
        for (int i = 0; i < repeat; i++) fn();
        double dt = (now_ns() - t0) / (double)repeat;
        if (dt < best) best = dt;
    }
    return best;
}

static size_t g_K;
static double *g_rio_re_hand, *g_rio_im_hand;
static double *g_rio_re_topo, *g_rio_im_topo;
static double *g_tw_re, *g_tw_im;
static double *g_in_orig_re, *g_in_orig_im;

static void call_hand(void) {
    radix16_t1_dit_fwd_avx512(g_rio_re_hand, g_rio_im_hand, g_tw_re, g_tw_im, g_K, g_K);
}
static void call_topo(void) {
    radix16_t1_dit_fwd_avx512_gen_inplace(g_rio_re_topo, g_rio_im_topo, g_tw_re, g_tw_im, g_K, g_K);
}

int main(int argc, char **argv) {
    size_t K = (argc > 1) ? (size_t)atoi(argv[1]) : 1024;
    if (K < 8 || K % 8 != 0) {
        fprintf(stderr, "K must be a multiple of 8\n"); return 1;
    }
    g_K = K;

    /* R=16: 16 legs × K elements per buffer; 15 twiddles × K. */
    g_in_orig_re   = aligned_alloc_doubles(16 * K);
    g_in_orig_im   = aligned_alloc_doubles(16 * K);
    g_rio_re_hand  = aligned_alloc_doubles(16 * K);
    g_rio_im_hand  = aligned_alloc_doubles(16 * K);
    g_rio_re_topo  = aligned_alloc_doubles(16 * K);
    g_rio_im_topo  = aligned_alloc_doubles(16 * K);
    g_tw_re        = aligned_alloc_doubles(15 * K);
    g_tw_im        = aligned_alloc_doubles(15 * K);

    fill_random(g_in_orig_re, 16 * K, 0xa1);
    fill_random(g_in_orig_im, 16 * K, 0xa2);
    fill_random(g_tw_re, 15 * K, 0xb1);
    fill_random(g_tw_im, 15 * K, 0xb2);

    memcpy(g_rio_re_hand, g_in_orig_re, 16 * K * sizeof(double));
    memcpy(g_rio_im_hand, g_in_orig_im, 16 * K * sizeof(double));
    memcpy(g_rio_re_topo, g_in_orig_re, 16 * K * sizeof(double));
    memcpy(g_rio_im_topo, g_in_orig_im, 16 * K * sizeof(double));

    call_hand();
    call_topo();

    double err_re = max_rel_err(g_rio_re_hand, g_rio_re_topo, 16 * K);
    double err_im = max_rel_err(g_rio_im_hand, g_rio_im_topo, 16 * K);
    double err = err_re > err_im ? err_re : err_im;

    if (err > 1e-10) {
        printf("CORRECTNESS FAIL: max_rel_err = %.2e\n", err);
        printf("  re: %.2e  im: %.2e\n", err_re, err_im);
        return 2;
    }

    int repeat = 5000;
    int trials = 7;

    double t_hand = bench(call_hand, repeat, trials);
    double t_topo = bench(call_topo, repeat, trials);

    printf("K=%zu  Hand=%.1fns  Topo=%.1fns  |  Topo/Hand=%.3f\n",
           K, t_hand, t_topo, t_topo / t_hand);
    return 0;
}
