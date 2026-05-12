/* bench_dct3_n8.c — DCT-III N=8 head-to-head:
 *   Production hand-written codelet  vs  OCaml DAG compiler emit.
 *
 * Production has the specialized N=8 codelet but the general-N
 * dispatcher is deferred per src/core/dct.h v1.0 note. So the OCaml
 * port also fills a real production gap at N != 8.
 *
 * Also validates correctness as "DCT-III(DCT-II(x)) = 2N · x" (FFTW
 * unnormalized inverse).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

#define N 8

#include "../../../vectorfft_tune/generated/dct8/dct3_n8_avx2.h"
#include "../../../vectorfft_tune/generated/dct8/dct2_n8_avx2.h"

__attribute__((target("avx2,fma")))
void radix8_dct3_avx2_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im,
    size_t K);

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
        if (fabs(b[i]) > scale) scale = fabs(b[i]);
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

static size_t g_K;
static double *g_in;
static double *g_out_prod;
static double *g_out_ocaml_re;
static double *g_out_ocaml_im;
static double *g_dummy;

static void call_prod(void) {
    dct3_n8_avx2(g_in, g_out_prod, g_K);
}
static void call_ocaml(void) {
    radix8_dct3_avx2_gen(g_in, g_dummy,
                         g_out_ocaml_re, g_out_ocaml_im,
                         g_dummy, g_dummy, g_K);
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

/* Roundtrip identity check: x' = dct3_ocaml(dct2_prod(x)) / (2N). */
static int roundtrip_check(size_t K) {
    double *x      = aa(N * K);
    double *y      = aa(N * K);
    double *z_re   = aa(N * K);
    double *z_im   = aa(N * K);
    double *dummy  = aa(N * K);
    fr(x, N * K, 0xDC);
    dct2_n8_avx2(x, y, K);
    radix8_dct3_avx2_gen(y, dummy, z_re, z_im, dummy, dummy, K);
    /* Normalize: x_recovered = z / (2N) */
    double scale = 1.0 / (2.0 * (double)N);
    double err = 0;
    for (size_t i = 0; i < N * K; i++) {
        double d = fabs(z_re[i] * scale - x[i]);
        if (d > err) err = d;
    }
    free(x); free(y); free(z_re); free(z_im); free(dummy);
    return err < 1e-12 ? 0 : 1;
}

static int run_one(size_t K) {
    g_K = K;
    g_in            = aa(N * K);
    g_out_prod      = aa(N * K);
    g_out_ocaml_re  = aa(N * K);
    g_out_ocaml_im  = aa(N * K);
    g_dummy         = aa(N * K);

    fr(g_in, N * K, 0xC3);
    memset(g_dummy, 0, N * K * sizeof(double));

    call_prod();
    call_ocaml();
    double err = max_rel(g_out_prod, g_out_ocaml_re, N * K);
    int pass = (err < 1e-10);
    if (!pass) {
        printf("K=%-5zu  CORRECTNESS FAIL  max_rel_err = %.2e\n", K, err);
        free(g_in); free(g_out_prod); free(g_out_ocaml_re);
        free(g_out_ocaml_im); free(g_dummy);
        return 1;
    }

    int repeat = (K <= 64) ? 20000 : (K <= 256 ? 5000 : 1000);
    int trials = 7;
    double t_prod  = bench(call_prod,  repeat, trials);
    double t_ocaml = bench(call_ocaml, repeat, trials);
    double ratio = t_ocaml / t_prod;
    const char *verdict;
    if (ratio < 0.98)      verdict = "OCaml WINS";
    else if (ratio < 1.05) verdict = "TIE";
    else if (ratio < 1.15) verdict = "OCaml SLOWER";
    else                   verdict = "REGRESSION";
    printf("K=%-5zu  Prod=%7.1f ns  OCaml=%7.1f ns  ratio=%5.3f  err=%.1e  %s\n",
           K, t_prod, t_ocaml, ratio, err, verdict);

    free(g_in); free(g_out_prod); free(g_out_ocaml_re);
    free(g_out_ocaml_im); free(g_dummy);
    return 0;
}

int main(void) {
    printf("================================================================\n");
    printf("  DCT-III N=8 bench: production hand-written vs OCaml DAG compiler\n");
    printf("================================================================\n");
    printf("  Op count (vector instructions):\n");
    printf("    Production dct3_n8_avx2:    29  [bench target]\n");
    printf("    OCaml radix8_dct3_avx2_gen: 58  (+100%% static; binary FMA-fuses)\n");
    printf("================================================================\n");
    printf("Roundtrip identity check (dct3∘dct2)/2N = x:\n");
    int rt = roundtrip_check(32);
    printf("  K=32  %s\n", rt == 0 ? "PASS" : "FAIL");
    if (rt) { printf("\nDCT-III implementation has a bug — bench aborted.\n"); return 1; }
    printf("\n");
    size_t Ks[] = {32, 64, 128, 256, 512, 1024, 0};
    int fails = 0;
    for (int i = 0; Ks[i] != 0; i++) fails += run_one(Ks[i]);
    printf("\n%s\n", fails == 0 ? "All correctness PASS" : "Some FAIL");
    return fails;
}
