/* bench_r4_avx2.c — R=4 OCaml vs Python (production) hand-coded codelet.
 *
 * Both compute a twiddled DIT-4 codelet on K parallel batches in-place.
 * Python takes (in, out, tw, K) but works in-place when in==out.
 * OCaml takes (rio, tw, ios, me) with ios=me=K.
 *
 * Note: at R=4 the OCaml cost model does NOT auto-apply the SU+spill
 * recipe (the codelet is too small for it to help — vec_regs > 4*2+6).
 * So the function name is plain "..._gen_inplace" without "_su_spill".
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

/* Hand-coded Python R=4 (header included as static functions) */
#include "r4_hand_avx2.h"

/* OCaml-generated R=4 — extern */
__attribute__((target("avx2,fma")))
void radix4_t1_dit_fwd_avx2_gen_inplace(
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

static void call_hand(void) {
    radix4_tw_dit_kernel_fwd_avx2(
        g_rio_re_a, g_rio_im_a,    /* in_re, in_im */
        g_rio_re_a, g_rio_im_a,    /* out_re, out_im — alias for in-place */
        g_tw_re, g_tw_im, g_K);
}

static void call_ocaml(void) {
    radix4_t1_dit_fwd_avx2_gen_inplace(
        g_rio_re_b, g_rio_im_b, g_tw_re, g_tw_im, g_K, g_K);
}

static int run_one(size_t K) {
    g_K = K;
    g_in_orig_re = aa(4 * K);
    g_in_orig_im = aa(4 * K);
    g_rio_re_a   = aa(4 * K);
    g_rio_im_a   = aa(4 * K);
    g_rio_re_b   = aa(4 * K);
    g_rio_im_b   = aa(4 * K);
    g_tw_re      = aa(3 * K);
    g_tw_im      = aa(3 * K);

    fr(g_in_orig_re, 4 * K, 0xa1);
    fr(g_in_orig_im, 4 * K, 0xa2);
    fr(g_tw_re, 3 * K, 0xb1);
    fr(g_tw_im, 3 * K, 0xb2);

    memcpy(g_rio_re_a, g_in_orig_re, 4 * K * sizeof(double));
    memcpy(g_rio_im_a, g_in_orig_im, 4 * K * sizeof(double));
    memcpy(g_rio_re_b, g_in_orig_re, 4 * K * sizeof(double));
    memcpy(g_rio_im_b, g_in_orig_im, 4 * K * sizeof(double));

    call_hand();
    call_ocaml();
    double err_re = max_rel(g_rio_re_a, g_rio_re_b, 4 * K);
    double err_im = max_rel(g_rio_im_a, g_rio_im_b, 4 * K);
    double err = err_re > err_im ? err_re : err_im;
    if (err >= 1e-10) {
        printf("R=4   K=%-4zu  CORRECTNESS FAIL  max_rel_err = %.2e\n", K, err);
        return 1;
    }

    memcpy(g_rio_re_a, g_in_orig_re, 4 * K * sizeof(double));
    memcpy(g_rio_im_a, g_in_orig_im, 4 * K * sizeof(double));
    memcpy(g_rio_re_b, g_in_orig_re, 4 * K * sizeof(double));
    memcpy(g_rio_im_b, g_in_orig_im, 4 * K * sizeof(double));

    /* R=4 is tiny — high repeat count */
    int repeat = 20000;
    int trials = 7;
    double t_hand  = bench(call_hand,  repeat, trials);
    double t_ocaml = bench(call_ocaml, repeat, trials);
    double ratio = t_ocaml / t_hand;
    const char *verdict;
    if (ratio < 0.98)      verdict = "OCaml WINS";
    else if (ratio < 1.05) verdict = "TIE";
    else if (ratio < 1.15) verdict = "OCaml SLOWER";
    else                   verdict = "REGRESSION";

    printf("R=4   K=%-4zu  Hand=%7.1f ns  OCaml=%7.1f ns  ratio=%5.3f  %s\n",
           K, t_hand, t_ocaml, ratio, verdict);

    free(g_in_orig_re); free(g_in_orig_im);
    free(g_rio_re_a); free(g_rio_im_a);
    free(g_rio_re_b); free(g_rio_im_b);
    free(g_tw_re); free(g_tw_im);
    return 0;
}

int main(int argc, char **argv) {
    size_t Ks_default[] = {64, 128, 256, 512, 1024, 2048, 4096, 0};
    size_t *Ks = Ks_default;
    size_t single[2] = {0, 0};
    if (argc > 1) { single[0] = (size_t)atoi(argv[1]); Ks = single; }

    printf("================================================================\n");
    printf("  Regression bench (AVX2): R=4 -- Hand (Python) vs OCaml\n");
    printf("================================================================\n");

    int fails = 0;
    for (int i = 0; Ks[i] != 0; i++) {
        if (Ks[i] % 4) continue;
        fails += run_one(Ks[i]);
    }
    printf("\n%s\n", fails == 0 ? "All correctness checks PASSED"
                                : "Some checks FAILED");
    return fails;
}
