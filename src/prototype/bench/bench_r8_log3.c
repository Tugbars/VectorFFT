/* bench_r8.c — Correctness verification + timing comparison for R=8.
 *
 * Compares:
 *   A) hand_t1_dit  — user's hand-coded radix8_t1_dit_fwd_avx512 (in-place)
 *   B) gen_t1_dit   — generated radix8_t1_dit_log3_fwd_avx512_gen_inplace (in-place)
 *
 * Same methodology as bench_r4.c but with 8 legs instead of 4 and 7 twiddles.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

#include "../radix8_handcoded.h"

__attribute__((target("avx512f")))
void radix8_t1_dit_log3_fwd_avx512_gen_inplace(
    double       * __restrict__ rio_re,
    double       * __restrict__ rio_im,
    const double * __restrict__ tw_re,
    const double * __restrict__ tw_im,
    size_t ios,
    size_t me);

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
static double *g_rio_re_hand,  *g_rio_im_hand;   /* 8*K doubles per buffer */
static double *g_rio_re_gen,   *g_rio_im_gen;
static double *g_tw_re,        *g_tw_im;          /* 7*K twiddle entries */
static double *g_in_orig_re,   *g_in_orig_im;

static void call_handcoded(void) {
    radix8_t1_dit_log3_fwd_avx512(g_rio_re_hand, g_rio_im_hand,
                             g_tw_re, g_tw_im,
                             g_K, g_K);
}

static void call_generated(void) {
    radix8_t1_dit_log3_fwd_avx512_gen_inplace(g_rio_re_gen, g_rio_im_gen,
                                          g_tw_re, g_tw_im,
                                          g_K, g_K);
}

int main(int argc, char **argv) {
    size_t K = (argc > 1) ? (size_t)atoi(argv[1]) : 1024;
    if (K < 8 || K % 8 != 0) {
        fprintf(stderr, "K must be a multiple of 8\n"); return 1;
    }
    g_K = K;

    /* R=8: 8 legs × K elements per buffer; 7 twiddles × K. */
    g_in_orig_re   = aligned_alloc_doubles(8 * K);
    g_in_orig_im   = aligned_alloc_doubles(8 * K);
    g_rio_re_hand  = aligned_alloc_doubles(8 * K);
    g_rio_im_hand  = aligned_alloc_doubles(8 * K);
    g_rio_re_gen   = aligned_alloc_doubles(8 * K);
    g_rio_im_gen   = aligned_alloc_doubles(8 * K);
    g_tw_re        = aligned_alloc_doubles(7 * K);
    g_tw_im        = aligned_alloc_doubles(7 * K);

    fill_random(g_in_orig_re, 8 * K, 0xa1);
    fill_random(g_in_orig_im, 8 * K, 0xa2);
    fill_random(g_tw_re, 7 * K, 0xb1);
    fill_random(g_tw_im, 7 * K, 0xb2);

    /* Correctness */
    memcpy(g_rio_re_hand, g_in_orig_re, 8 * K * sizeof(double));
    memcpy(g_rio_im_hand, g_in_orig_im, 8 * K * sizeof(double));
    memcpy(g_rio_re_gen,  g_in_orig_re, 8 * K * sizeof(double));
    memcpy(g_rio_im_gen,  g_in_orig_im, 8 * K * sizeof(double));

    radix8_t1_dit_log3_fwd_avx512(g_rio_re_hand, g_rio_im_hand, g_tw_re, g_tw_im, K, K);
    radix8_t1_dit_log3_fwd_avx512_gen_inplace(g_rio_re_gen, g_rio_im_gen, g_tw_re, g_tw_im, K, K);

    double err_re = max_rel_err(g_rio_re_hand, g_rio_re_gen, 8 * K);
    double err_im = max_rel_err(g_rio_im_hand, g_rio_im_gen, 8 * K);
    double max_err = err_re > err_im ? err_re : err_im;

    printf("=== Correctness ===\n");
    printf("  K = %zu\n", K);
    printf("  max_rel_err re = %.2e\n", err_re);
    printf("  max_rel_err im = %.2e\n", err_im);
    printf("  status: %s\n\n", max_err < 1e-10 ? "OK" : "MISMATCH");

    if (max_err >= 1e-10) {
        printf("First few mismatches (re):\n");
        int shown = 0;
        for (size_t i = 0; i < 8*K && shown < 8; i++) {
            if (fabs(g_rio_re_hand[i] - g_rio_re_gen[i]) > 1e-10) {
                printf("  [%zu] hand=%.6f gen=%.6f diff=%.2e\n",
                       i, g_rio_re_hand[i], g_rio_re_gen[i],
                       g_rio_re_hand[i] - g_rio_re_gen[i]);
                shown++;
            }
        }
        return 2;
    }

    /* Timing */
    int repeat = 5000;
    int trials = 7;

    double t_hand = bench(call_handcoded, repeat, trials);
    double t_gen  = bench(call_generated, repeat, trials);

    printf("=== Timing ===\n");
    printf("  K = %zu, repeat = %d, trials = %d, taking min\n", K, repeat, trials);
    printf("  hand-coded:  %8.1f ns/call\n", t_hand);
    printf("  generated:   %8.1f ns/call\n", t_gen);
    printf("  ratio gen/hand: %.3f\n", t_gen / t_hand);
    return 0;
}
