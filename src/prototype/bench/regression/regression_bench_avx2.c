/* regression_bench_avx2.c — AVX2 counterpart of regression_bench.c.
 * Compares hand-coded (Python) vs OCaml-generated codelets for R=16, 25,
 * 32, 64 on AVX2. Same harness structure as the AVX-512 version; only
 * the target attribute and SIMD lane width differ.
 *
 * Note on lane widths:
 *   AVX-512: 8 doubles per ZMM, codelet inner loop steps k += 8
 *   AVX2:    4 doubles per YMM, codelet inner loop steps k += 4
 * Both bench at K ∈ {64, 128, 256, 512, 1024} which are multiples of 8
 * (so safe for both ISAs).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

/* Hand-coded AVX2 codelets — included as static (target avx2,fma) */
#include "r16_hand_avx2.h"
#include "r25_hand_avx2.h"
#include "r32_hand_avx2.h"
#include "r64_hand_avx2.h"

/* OCaml-generated AVX2 codelets — externs */
__attribute__((target("avx2,fma")))
void radix16_t1_dit_fwd_avx2(
    double *, double *, const double *, const double *, size_t, size_t);

__attribute__((target("avx2,fma")))
void radix25_t1_dit_fwd_avx2(
    double *, double *, const double *, const double *, size_t, size_t);

__attribute__((target("avx2,fma")))
void radix32_t1_dit_fwd_avx2(
    double *, double *, const double *, const double *, size_t, size_t);

__attribute__((target("avx2,fma")))
void radix64_t1_dit_fwd_avx2(
    double *, double *, const double *, const double *, size_t, size_t);

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

static double max_rel(const double *a, const double *b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        double scale = fabs(a[i]);
        double sb = fabs(b[i]);
        if (sb > scale) scale = sb;
        if (scale < 1e-3) scale = 1e-3;
        double r = d / scale;
        if (r > m) m = r;
    }
    return m;
}

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
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
static double *g_rio_re_a, *g_rio_im_a;
static double *g_rio_re_b, *g_rio_im_b;
static double *g_tw_re, *g_tw_im;
static double *g_in_orig_re, *g_in_orig_im;

static void call_hand_r16(void) {
    radix16_t1_dit_fwd_avx2(g_rio_re_a, g_rio_im_a, g_tw_re, g_tw_im, g_K, g_K);
}
static void call_ocaml_r16(void) {
    radix16_t1_dit_fwd_avx2(
        g_rio_re_b, g_rio_im_b, g_tw_re, g_tw_im, g_K, g_K);
}

static void call_hand_r25(void) {
    radix25_t1_dit_fwd_avx2(g_rio_re_a, g_rio_im_a, g_tw_re, g_tw_im, g_K, g_K);
}
static void call_ocaml_r25(void) {
    radix25_t1_dit_fwd_avx2(
        g_rio_re_b, g_rio_im_b, g_tw_re, g_tw_im, g_K, g_K);
}

static void call_hand_r32(void) {
    radix32_t1_dit_fwd_avx2(g_rio_re_a, g_rio_im_a, g_tw_re, g_tw_im, g_K, g_K);
}
static void call_ocaml_r32(void) {
    radix32_t1_dit_fwd_avx2(
        g_rio_re_b, g_rio_im_b, g_tw_re, g_tw_im, g_K, g_K);
}

static void call_hand_r64(void) {
    radix64_t1_dit_fwd_avx2(g_rio_re_a, g_rio_im_a, g_tw_re, g_tw_im, g_K, g_K);
}
static void call_ocaml_r64(void) {
    radix64_t1_dit_fwd_avx2(
        g_rio_re_b, g_rio_im_b, g_tw_re, g_tw_im, g_K, g_K);
}

static int run_one(int R, int n_tw,
                   void (*hand)(void), void (*ocaml)(void),
                   size_t K) {
    g_K = K;
    g_in_orig_re = aa((size_t)R * K);
    g_in_orig_im = aa((size_t)R * K);
    g_rio_re_a   = aa((size_t)R * K);
    g_rio_im_a   = aa((size_t)R * K);
    g_rio_re_b   = aa((size_t)R * K);
    g_rio_im_b   = aa((size_t)R * K);
    g_tw_re      = aa((size_t)n_tw * K);
    g_tw_im      = aa((size_t)n_tw * K);

    fr(g_in_orig_re, (size_t)R * K, 0xa1 + R);
    fr(g_in_orig_im, (size_t)R * K, 0xa2 + R);
    fr(g_tw_re, (size_t)n_tw * K, 0xb1 + R);
    fr(g_tw_im, (size_t)n_tw * K, 0xb2 + R);

    memcpy(g_rio_re_a, g_in_orig_re, (size_t)R * K * sizeof(double));
    memcpy(g_rio_im_a, g_in_orig_im, (size_t)R * K * sizeof(double));
    memcpy(g_rio_re_b, g_in_orig_re, (size_t)R * K * sizeof(double));
    memcpy(g_rio_im_b, g_in_orig_im, (size_t)R * K * sizeof(double));

    hand();
    ocaml();
    double err_re = max_rel(g_rio_re_a, g_rio_re_b, (size_t)R * K);
    double err_im = max_rel(g_rio_im_a, g_rio_im_b, (size_t)R * K);
    double err = err_re > err_im ? err_re : err_im;

    int pass = (err < 1e-10);
    if (!pass) {
        printf("R=%-3d K=%-4zu  CORRECTNESS FAIL  max_rel_err = %.2e\n", R, K, err);
        return 1;
    }

    memcpy(g_rio_re_a, g_in_orig_re, (size_t)R * K * sizeof(double));
    memcpy(g_rio_im_a, g_in_orig_im, (size_t)R * K * sizeof(double));
    memcpy(g_rio_re_b, g_in_orig_re, (size_t)R * K * sizeof(double));
    memcpy(g_rio_im_b, g_in_orig_im, (size_t)R * K * sizeof(double));

    int repeat = 5000;
    int trials = 7;
    double t_hand  = bench(hand,  repeat, trials);
    double t_ocaml = bench(ocaml, repeat, trials);

    double ratio = t_ocaml / t_hand;
    const char *verdict;
    if (ratio < 0.98)      verdict = "OCaml WINS";
    else if (ratio < 1.05) verdict = "TIE";
    else if (ratio < 1.15) verdict = "OCaml SLOWER";
    else                   verdict = "REGRESSION";

    printf("R=%-3d K=%-4zu  Hand=%7.1f ns  OCaml=%7.1f ns  ratio=%5.3f  %s\n",
           R, K, t_hand, t_ocaml, ratio, verdict);

    free(g_in_orig_re); free(g_in_orig_im);
    free(g_rio_re_a); free(g_rio_im_a);
    free(g_rio_re_b); free(g_rio_im_b);
    free(g_tw_re); free(g_tw_im);
    return 0;
}

int main(int argc, char **argv) {
    size_t Ks_default[] = {64, 128, 256, 512, 1024, 0};
    size_t *Ks = Ks_default;
    size_t single_K[2] = {0, 0};
    if (argc > 1) {
        single_K[0] = (size_t)atoi(argv[1]);
        Ks = single_K;
    }

    printf("════════════════════════════════════════════════════════════════════════\n");
    printf("  Regression bench (AVX2): Hand (Python) vs OCaml (vfft_v2)\n");
    printf("  OCaml config: default (recipe + SU + spill auto-fire for R≥5)\n");
    printf("  Compiler:     gcc-11 -O3 -flive-range-shrinkage -mavx2 -mfma\n");
    printf("  SIMD width:   4 doubles per YMM (vs 8 per ZMM on AVX-512)\n");
    printf("════════════════════════════════════════════════════════════════════════\n");

    int total_fails = 0;
    for (int i = 0; Ks[i] != 0; i++) {
        size_t K = Ks[i];
        if (K < 4 || K % 4 != 0) continue;
        printf("\n--- K = %zu ---\n", K);
        total_fails += run_one(16, 15, call_hand_r16, call_ocaml_r16, K);
        total_fails += run_one(25, 24, call_hand_r25, call_ocaml_r25, K);
        total_fails += run_one(32, 31, call_hand_r32, call_ocaml_r32, K);
        total_fails += run_one(64, 63, call_hand_r64, call_ocaml_r64, K);
    }

    printf("\n════════════════════════════════════════════════════════════════════════\n");
    if (total_fails == 0) printf("  All correctness checks PASSED\n");
    else                  printf("  %d configurations FAILED correctness\n", total_fails);
    printf("════════════════════════════════════════════════════════════════════════\n");
    return total_fails;
}
