/* bench_dst_n8.c — DST-II and DST-III at N=8: OCaml fused DAG codelet
 *                   vs production-style 3-pass (sign-flip + DCT + reverse).
 *
 * Production's `src/core/dst.h` does this at runtime:
 *   DST-II:  prebuf[n] = (-1)^n · re[n]  → dct2_n8_avx2 → out[k] = prebuf[N-1-k]
 *   DST-III: prebuf = reverse(re) → dct3_n8_avx2 → out[k] = (-1)^k · prebuf[k]
 *
 * Our fused DAG codelet does the whole thing in one call. */

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

#include "../../../vectorfft_tune/generated/dct8/dct2_n8_avx2.h"
#include "../../../vectorfft_tune/generated/dct8/dct3_n8_avx2.h"

__attribute__((target("avx2,fma"))) void radix8_dst2_avx2_gen(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
__attribute__((target("avx2,fma"))) void radix8_dst3_avx2_gen(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);

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

/* Brute-force DST-II */
static void bf_dst2(const double *x, double *out, size_t K) {
    double xb[N];
    for (int n = 0; n < N; n++) xb[n] = x[(size_t)n * K + 0];
    for (int k = 0; k < N; k++) {
        double s = 0;
        for (int n = 0; n < N; n++) {
            double t = M_PI * (k + 1) * (2 * n + 1) / (2.0 * N);
            s += 2.0 * xb[n] * sin(t);
        }
        out[k] = s;
    }
}

/* Brute-force DST-III */
static void bf_dst3(const double *x, double *out, size_t K) {
    double xb[N];
    for (int n = 0; n < N; n++) xb[n] = x[(size_t)n * K + 0];
    for (int k = 0; k < N; k++) {
        double s = ((k & 1) ? -1.0 : 1.0) * xb[N - 1];
        for (int n = 0; n < N - 1; n++) {
            double t = M_PI * (n + 1) * (2 * k + 1) / (2.0 * N);
            s += 2.0 * xb[n] * sin(t);
        }
        out[k] = s;
    }
}

/* Production-style 3-pass DST-II. */
__attribute__((target("avx2,fma")))
static void dst2_3pass(const double *in, double *out, double *prebuf, size_t K) {
    /* Phase 1: sign-flip every other row */
    for (int n = 0; n < N; n++) {
        const double *src = in     + (size_t)n * K;
        double       *dst = prebuf + (size_t)n * K;
        if (n & 1) {
            for (size_t k = 0; k < K; k += 4) {
                __m256d v = _mm256_loadu_pd(src + k);
                _mm256_storeu_pd(dst + k,
                    _mm256_xor_pd(v, _mm256_set1_pd(-0.0)));
            }
        } else {
            memcpy(dst, src, K * sizeof(double));
        }
    }
    /* Phase 2: DCT-II */
    dct2_n8_avx2(prebuf, prebuf, K);
    /* Phase 3: reverse along N axis */
    for (int k = 0; k < N; k++) {
        const double *src = prebuf + (size_t)(N - 1 - k) * K;
        double       *dst = out    + (size_t)k * K;
        memcpy(dst, src, K * sizeof(double));
    }
}

/* Production-style 3-pass DST-III. */
__attribute__((target("avx2,fma")))
static void dst3_3pass(const double *in, double *out, double *prebuf, size_t K) {
    /* Phase 1: reverse along N axis */
    for (int n = 0; n < N; n++) {
        const double *src = in     + (size_t)(N - 1 - n) * K;
        double       *dst = prebuf + (size_t)n * K;
        memcpy(dst, src, K * sizeof(double));
    }
    /* Phase 2: DCT-III */
    dct3_n8_avx2(prebuf, prebuf, K);
    /* Phase 3: sign-flip every other output row */
    for (int k = 0; k < N; k++) {
        const double *src = prebuf + (size_t)k * K;
        double       *dst = out    + (size_t)k * K;
        if (k & 1) {
            for (size_t b = 0; b < K; b += 4) {
                __m256d v = _mm256_loadu_pd(src + b);
                _mm256_storeu_pd(dst + b,
                    _mm256_xor_pd(v, _mm256_set1_pd(-0.0)));
            }
        } else {
            memcpy(dst, src, K * sizeof(double));
        }
    }
}

static double bench_fn(void (*fn)(void), int repeat, int trials) {
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
static double *g_in, *g_out_3pass, *g_out_fused, *g_out_im, *g_prebuf, *g_dummy;
static int    g_mode; /* 0=DST-II, 1=DST-III */

static void call_3pass(void) {
    if (g_mode == 0)
        dst2_3pass(g_in, g_out_3pass, g_prebuf, g_K);
    else
        dst3_3pass(g_in, g_out_3pass, g_prebuf, g_K);
}
static void call_fused(void) {
    if (g_mode == 0)
        radix8_dst2_avx2_gen(g_in, g_dummy, g_out_fused, g_out_im,
                             g_dummy, g_dummy, g_K);
    else
        radix8_dst3_avx2_gen(g_in, g_dummy, g_out_fused, g_out_im,
                             g_dummy, g_dummy, g_K);
}

static int run_one(int mode, size_t K) {
    g_mode = mode;
    g_K = K;
    g_in         = aa(N * K);
    g_out_3pass  = aa(N * K);
    g_out_fused  = aa(N * K);
    g_out_im     = aa(N * K);
    g_prebuf     = aa(N * K);
    g_dummy      = aa(N * K);
    fr(g_in, N * K, 0xD5u + (unsigned)mode);
    memset(g_dummy, 0, N * K * sizeof(double));

    call_3pass();
    call_fused();
    double ref[N];
    if (mode == 0) bf_dst2(g_in, ref, K);
    else           bf_dst3(g_in, ref, K);
    double err_3pass = 0, err_fused = 0;
    for (int k = 0; k < N; k++) {
        double v3 = g_out_3pass[(size_t)k * K + 0];
        double vf = g_out_fused[(size_t)k * K + 0];
        double r3 = fabs(v3 - ref[k]);
        double rf = fabs(vf - ref[k]);
        if (r3 > err_3pass) err_3pass = r3;
        if (rf > err_fused) err_fused = rf;
    }
    int ok = (err_3pass < 1e-10) && (err_fused < 1e-10);
    if (!ok) {
        printf("%s K=%-5zu  FAIL  3pass_err=%.2e  fused_err=%.2e\n",
               mode == 0 ? "DST-II" : "DST-III", K, err_3pass, err_fused);
        return 1;
    }

    int repeat = (K <= 64) ? 20000 : (K <= 256 ? 5000 : 1000);
    int trials = 7;
    double t3 = bench_fn(call_3pass, repeat, trials);
    double tf = bench_fn(call_fused, repeat, trials);
    double ratio = tf / t3;
    const char *verdict;
    if (ratio < 0.98)      verdict = "fused WINS";
    else if (ratio < 1.05) verdict = "TIE";
    else if (ratio < 1.15) verdict = "fused SLOWER";
    else                   verdict = "REGRESSION";
    printf("%s K=%-5zu  3pass=%7.1f ns  fused=%7.1f ns  ratio=%5.3f  %s\n",
           mode == 0 ? "DST-II " : "DST-III", K, t3, tf, ratio, verdict);
    return 0;
}

int main(void) {
    printf("================================================================\n");
    printf("  DST-II/III N=8: OCaml fused DAG vs production-style 3-pass\n");
    printf("================================================================\n");
    printf("  Op counts (vector instructions):\n");
    printf("    DST-II  fused: 48  (sign-flip + DCT-II + reverse absorbed)\n");
    printf("    DST-III fused: 55  (reverse + DCT-III + sign-flip absorbed)\n");
    printf("    3-pass: ~36 (DCT-II/III hand codelet) + 2 memory passes\n");
    printf("================================================================\n");
    size_t Ks[] = {32, 64, 128, 256, 512, 1024, 0};
    int fails = 0;
    printf("\n-- DST-II --\n");
    for (int i = 0; Ks[i] != 0; i++) fails += run_one(0, Ks[i]);
    printf("\n-- DST-III --\n");
    for (int i = 0; Ks[i] != 0; i++) fails += run_one(1, Ks[i]);
    printf("\n%s\n", fails == 0 ? "All correctness PASS" : "Some FAIL");
    return fails;
}
