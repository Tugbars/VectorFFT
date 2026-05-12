/* bench_dht_n8.c — DHT N=8 head-to-head: OCaml fused DAG codelet vs
 *                  the 3-pass approach production uses (R2C + butterfly).
 *
 * Production has no specialized DHT codelet at any N, so the comparison
 * is OCaml fused codelet vs production-style 3-pass:
 *   3-pass:    radix8_rdft_fwd_avx2_gen  +  hand-written butterfly
 *   fused:     radix8_dht_avx2_gen                                       */

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

#define N 8

__attribute__((target("avx2,fma")))
void radix8_dht_avx2_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im,
    size_t K);

__attribute__((target("avx2,fma")))
void radix8_rdft_fwd_avx2_gen(
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

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

static double max_abs(const double *a, const double *b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

/* Brute-force DHT: H[k] = sum x[n] · (cos + sin)(2πkn/N), batch lane 0 only.
 * Out is a contiguous [N] array (not batched). */
static void bf_dht(const double *x, double *out, size_t K) {
    double xb[N];
    for (int n = 0; n < N; n++) xb[n] = x[(size_t)n * K + 0];
    for (int k = 0; k < N; k++) {
        double s = 0;
        for (int n = 0; n < N; n++) {
            double t = 2.0 * M_PI * k * n / N;
            s += xb[n] * (cos(t) + sin(t));
        }
        out[k] = s;
    }
}

/* 3-pass production-style: rdft + hand butterfly. */
__attribute__((target("avx2,fma")))
static void dht_3pass(const double *in, double *out,
                      double *scratch_re, double *scratch_im,
                      const double *dummy, size_t K) {
    radix8_rdft_fwd_avx2_gen(in, dummy, scratch_re, scratch_im, dummy, dummy, K);
    for (size_t b = 0; b < K; b += 4) {
        __m256d re0 = _mm256_loadu_pd(&scratch_re[0*K + b]);
        _mm256_storeu_pd(&out[0*K + b], re0);
        __m256d re4 = _mm256_loadu_pd(&scratch_re[4*K + b]);
        _mm256_storeu_pd(&out[4*K + b], re4);
        for (int k = 1; k < 4; k++) {
            __m256d re = _mm256_loadu_pd(&scratch_re[k*K + b]);
            __m256d im = _mm256_loadu_pd(&scratch_im[k*K + b]);
            _mm256_storeu_pd(&out[k*K + b],     _mm256_sub_pd(re, im));
            _mm256_storeu_pd(&out[(N-k)*K + b], _mm256_add_pd(re, im));
        }
    }
}

static size_t g_K;
static double *g_in, *g_out_3pass, *g_out_ocaml_re, *g_out_ocaml_im;
static double *g_scratch_re, *g_scratch_im, *g_dummy;

static void call_3pass(void) {
    dht_3pass(g_in, g_out_3pass, g_scratch_re, g_scratch_im, g_dummy, g_K);
}
static void call_fused(void) {
    radix8_dht_avx2_gen(g_in, g_dummy, g_out_ocaml_re, g_out_ocaml_im,
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

static int run_one(size_t K) {
    g_K = K;
    g_in           = aa(N * K);
    g_out_3pass    = aa(N * K);
    g_out_ocaml_re = aa(N * K);
    g_out_ocaml_im = aa(N * K);
    g_scratch_re   = aa(N * K);
    g_scratch_im   = aa(N * K);
    g_dummy        = aa(N * K);
    fr(g_in, N * K, 0xDA);
    memset(g_dummy, 0, N * K * sizeof(double));

    call_3pass();
    call_fused();
    double err_3pass = 0, err_ocaml = 0;
    {
        double ref[N];
        bf_dht(g_in, ref, K);
        for (int k = 0; k < N; k++) {
            double v3 = g_out_3pass[(size_t)k * K + 0];
            double vo = g_out_ocaml_re[(size_t)k * K + 0];
            double d3 = fabs(v3 - g_in[(size_t)k * K + 0]);  /* dummy */
            (void)d3;
            double r3 = fabs(v3 - ref[k]);
            double ro = fabs(vo - ref[k]);
            if (r3 > err_3pass) err_3pass = r3;
            if (ro > err_ocaml) err_ocaml = ro;
        }
    }
    int pass = (err_3pass < 1e-10) && (err_ocaml < 1e-10);
    if (!pass) {
        printf("K=%-5zu  FAIL  3pass_err=%.2e  ocaml_err=%.2e\n",
               K, err_3pass, err_ocaml);
        return 1;
    }

    int repeat = (K <= 64) ? 20000 : (K <= 256 ? 5000 : 1000);
    int trials = 7;
    double t_3pass = bench(call_3pass, repeat, trials);
    double t_fused = bench(call_fused, repeat, trials);
    double ratio = t_fused / t_3pass;
    const char *verdict;
    if (ratio < 0.98)      verdict = "fused WINS";
    else if (ratio < 1.05) verdict = "TIE";
    else if (ratio < 1.15) verdict = "fused SLOWER";
    else                   verdict = "REGRESSION";
    printf("K=%-5zu  3pass=%7.1f ns  fused=%7.1f ns  ratio=%5.3f  %s\n",
           K, t_3pass, t_fused, ratio, verdict);
    return 0;
}

int main(void) {
    printf("================================================================\n");
    printf("  DHT N=8: OCaml fused DAG codelet vs 3-pass (rdft+butterfly)\n");
    printf("================================================================\n");
    printf("  Op count:\n");
    printf("    3-pass (rdft_8 + butterfly): ~33 (23 rdft + ~10 butterfly)\n");
    printf("    OCaml fused dht codelet:      28 (single fused DAG)\n");
    printf("================================================================\n");
    size_t Ks[] = {32, 64, 128, 256, 512, 1024, 0};
    int fails = 0;
    for (int i = 0; Ks[i] != 0; i++) fails += run_one(Ks[i]);
    return fails;
}
