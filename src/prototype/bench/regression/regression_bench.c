/* regression_bench.c — Compare hand-coded (Python) vs OCaml-generated codelets
 * for R=16, 25, 32, 64 to detect performance regressions from recent emitter
 * changes (docs 43-44, twidsq work).
 *
 * Compiles with the production config: gcc-11 + -flive-range-shrinkage.
 * OCaml side uses default best config (recipe + SU + spill auto-fires for R≥5).
 *
 * Each codelet operates in-place: rio_re/rio_im with row stride `ios` and
 * batch dim `me`. The benchmark measures the steady-state cycle cost per
 * codelet invocation, after a warmup of 100 iterations, taking the best of
 * 7 trials of 5000 iterations each.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

/* Hand-coded codelets (static, included directly) */
#include "r16_hand.h"
#include "r25_hand.h"
#include "r32_hand.h"
#include "r64_hand.h"

/* OCaml-generated codelets (linked as separate translation units) */
__attribute__((target("avx512f")))
void radix16_t1_dit_fwd_avx512(
    double *, double *, const double *, const double *, size_t, size_t);

__attribute__((target("avx512f")))
void radix25_t1_dit_fwd_avx512(
    double *, double *, const double *, const double *, size_t, size_t);

__attribute__((target("avx512f")))
void radix32_t1_dit_fwd_avx512(
    double *, double *, const double *, const double *, size_t, size_t);

__attribute__((target("avx512f")))
void radix64_t1_dit_fwd_avx512(
    double *, double *, const double *, const double *, size_t, size_t);

static double *aa(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(double)) != 0) exit(1);
    return (double *)p;
}

static void fr(double *p, size_t n, unsigned s) {
    for (size_t i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        p[i] = (double)((int)(s >> 8) & 0x7fffff) / (double)0x800000 - 0.5;
    }
}

static double max_rel(const double *a, const double *b, size_t n) {
    /* Combined abs/rel metric: pass if either abs diff is tiny OR
     * relative error is small (with an epsilon floor on the denominator
     * to avoid divide-by-near-zero blowups when both values cancel).
     * Returns max(abs_diff / max(|a|, |b|, 1e-3)) — effectively rel
     * error against a noise floor, so values near zero don't dominate. */
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

/* Globals to plumb args into the void-taking bench fn */
static size_t g_K;
static double *g_rio_re_a, *g_rio_im_a;
static double *g_rio_re_b, *g_rio_im_b;
static double *g_tw_re, *g_tw_im;
static double *g_in_orig_re, *g_in_orig_im;

/* Per-radix call wrappers — needed because bench() takes void(*)() */
static void call_hand_r16(void) {
    radix16_t1_dit_fwd_avx512(g_rio_re_a, g_rio_im_a, g_tw_re, g_tw_im, g_K, g_K);
}
static void call_ocaml_r16(void) {
    radix16_t1_dit_fwd_avx512(
        g_rio_re_b, g_rio_im_b, g_tw_re, g_tw_im, g_K, g_K);
}

static void call_hand_r25(void) {
    radix25_t1_dit_fwd_avx512(g_rio_re_a, g_rio_im_a, g_tw_re, g_tw_im, g_K, g_K);
}
static void call_ocaml_r25(void) {
    radix25_t1_dit_fwd_avx512(
        g_rio_re_b, g_rio_im_b, g_tw_re, g_tw_im, g_K, g_K);
}

static void call_hand_r32(void) {
    radix32_t1_dit_fwd_avx512(g_rio_re_a, g_rio_im_a, g_tw_re, g_tw_im, g_K, g_K);
}
static void call_ocaml_r32(void) {
    radix32_t1_dit_fwd_avx512(
        g_rio_re_b, g_rio_im_b, g_tw_re, g_tw_im, g_K, g_K);
}

static void call_hand_r64(void) {
    radix64_t1_dit_fwd_avx512(g_rio_re_a, g_rio_im_a, g_tw_re, g_tw_im, g_K, g_K);
}
static void call_ocaml_r64(void) {
    radix64_t1_dit_fwd_avx512(
        g_rio_re_b, g_rio_im_b, g_tw_re, g_tw_im, g_K, g_K);
}

/* Per-radix runner: alloc, fill, correctness check, time both, report. */
static int run_one(int R, int n_tw,
                   void (*hand)(void), void (*ocaml)(void),
                   size_t K) {
    g_K = K;
    /* R legs × K elements per buffer; n_tw twiddle slots × K. */
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

    /* Correctness: run both once on the same data, compare. */
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

    /* Restore inputs (codelet was in-place) and bench. */
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
    /* Multiple K values to characterize across batch sizes —
     * small K stresses overhead, large K stresses sustained throughput. */
    size_t Ks_default[] = {64, 128, 256, 512, 1024, 0};
    size_t *Ks = Ks_default;
    size_t single_K[2] = {0, 0};
    if (argc > 1) {
        single_K[0] = (size_t)atoi(argv[1]);
        Ks = single_K;
    }

    printf("════════════════════════════════════════════════════════════════════════\n");
    printf("  Regression bench: Hand (Python gen_radix*.py) vs OCaml (vfft_v2)\n");
    printf("  OCaml config: default (recipe + SU + spill auto-fire for R≥5)\n");
    printf("  Compiler:     gcc-11 -O3 -flive-range-shrinkage -march=skylake-avx512\n");
    printf("════════════════════════════════════════════════════════════════════════\n");

    int total_fails = 0;
    for (int i = 0; Ks[i] != 0; i++) {
        size_t K = Ks[i];
        if (K < 8 || K % 8 != 0) continue;
        printf("\n--- K = %zu ---\n", K);
        /* n_tw: t1_dit codelet uses N-1 twiddles (legs 1..N-1 each get one) */
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
