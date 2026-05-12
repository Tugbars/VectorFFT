/* bench_dht_scale.c — DHT at N=8, 16, 32, 64: fused DAG vs 3-pass.
 *
 * 3-pass mimics production's runtime:
 *   Phase 1: N-point rdft → (X_re, X_im) Hermitian-compact
 *   Phase 2: butterfly H[k] = Re(X[k]) ± Im(X[k])
 *
 * Validates the fused-vs-3-pass win generalizes beyond N=8. */
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

__attribute__((target("avx2,fma"))) void radix8_dht_avx2_gen(
    const double*, const double*, double*, double*,
    const double*, const double*, size_t);
__attribute__((target("avx2,fma"))) void radix16_dht_avx2_gen(
    const double*, const double*, double*, double*,
    const double*, const double*, size_t);
__attribute__((target("avx2,fma"))) void radix32_dht_avx2_gen(
    const double*, const double*, double*, double*,
    const double*, const double*, size_t);
__attribute__((target("avx2,fma"))) void radix64_dht_avx2_gen(
    const double*, const double*, double*, double*,
    const double*, const double*, size_t);

__attribute__((target("avx2,fma"))) void radix8_rdft_fwd_avx2_gen(
    const double*, const double*, double*, double*,
    const double*, const double*, size_t);
__attribute__((target("avx2,fma"))) void radix16_rdft_fwd_avx2_gen(
    const double*, const double*, double*, double*,
    const double*, const double*, size_t);
__attribute__((target("avx2,fma"))) void radix32_rdft_fwd_avx2_gen(
    const double*, const double*, double*, double*,
    const double*, const double*, size_t);
__attribute__((target("avx2,fma"))) void radix64_rdft_fwd_avx2_gen(
    const double*, const double*, double*, double*,
    const double*, const double*, size_t);

typedef void (*fn_t)(const double*, const double*, double*, double*,
                     const double*, const double*, size_t);

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

/* DHT 3-pass: rdft + butterfly. */
__attribute__((target("avx2,fma")))
static void dht_3pass(const double *in, double *out, int N, size_t K,
                      double *sre, double *sim,
                      fn_t rdft, const double *dummy) {
    rdft(in, dummy, sre, sim, dummy, dummy, K);
    int half = N / 2;
    for (size_t b = 0; b < K; b += 4) {
        __m256d r0 = _mm256_loadu_pd(sre + 0*K + b);
        _mm256_storeu_pd(out + 0*K + b, r0);
        __m256d rh = _mm256_loadu_pd(sre + (size_t)half * K + b);
        _mm256_storeu_pd(out + (size_t)half * K + b, rh);
    }
    for (int k = 1; k < half; k++) {
        size_t off_lo = (size_t)k * K;
        size_t off_hi = (size_t)(N - k) * K;
        for (size_t b = 0; b < K; b += 4) {
            __m256d re = _mm256_loadu_pd(sre + off_lo + b);
            __m256d im = _mm256_loadu_pd(sim + off_lo + b);
            _mm256_storeu_pd(out + off_lo + b, _mm256_sub_pd(re, im));
            _mm256_storeu_pd(out + off_hi + b, _mm256_add_pd(re, im));
        }
    }
}

typedef struct {
    int N;
    fn_t dht;
    fn_t rdft;
} cell_t;

static size_t g_K;
static cell_t *g_cell;
static double *g_in, *g_out_3pass, *g_out_fused, *g_out_im;
static double *g_sre, *g_sim, *g_dummy;

static void call_3pass(void) {
    dht_3pass(g_in, g_out_3pass, g_cell->N, g_K,
              g_sre, g_sim, g_cell->rdft, g_dummy);
}
static void call_fused(void) {
    g_cell->dht(g_in, g_dummy, g_out_fused, g_out_im,
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

static int run_one(cell_t *cell, size_t K) {
    int N = cell->N;
    g_cell = cell;
    g_K = K;
    g_in        = aa((size_t)N * K);
    g_out_3pass = aa((size_t)N * K);
    g_out_fused = aa((size_t)N * K);
    g_out_im    = aa((size_t)N * K);
    g_sre       = aa((size_t)N * K);
    g_sim       = aa((size_t)N * K);
    g_dummy     = aa((size_t)N * K);
    fr(g_in, (size_t)N * K, 0xDA + (unsigned)N);
    memset(g_dummy, 0, (size_t)N * K * sizeof(double));

    call_3pass();
    call_fused();
    double err = 0;
    for (size_t i = 0; i < (size_t)N * K; i++) {
        double d = fabs(g_out_3pass[i] - g_out_fused[i]);
        if (d > err) err = d;
    }
    int ok = err < 1e-10;
    if (!ok) {
        printf("N=%-3d K=%-5zu  FAIL  diff=%.2e\n", N, K, err);
        return 1;
    }

    int repeat = (K <= 64) ? 20000 : (K <= 256 ? 5000 : 1000);
    int trials = 7;
    double t3 = bench(call_3pass, repeat, trials);
    double tf = bench(call_fused, repeat, trials);
    double ratio = tf / t3;
    const char *verdict;
    if (ratio < 0.98)      verdict = "fused WINS";
    else if (ratio < 1.05) verdict = "TIE";
    else if (ratio < 1.15) verdict = "fused SLOWER";
    else                   verdict = "REGRESSION";
    printf("N=%-3d K=%-5zu  3pass=%8.1f ns  fused=%8.1f ns  ratio=%5.3f  %s\n",
           N, K, t3, tf, ratio, verdict);
    return 0;
}

int main(void) {
    printf("================================================================\n");
    printf("  DHT scaling: fused DAG vs 3-pass (rdft + butterfly) at N=8..64\n");
    printf("================================================================\n");
    cell_t cells[] = {
        {8,  radix8_dht_avx2_gen,  radix8_rdft_fwd_avx2_gen},
        {16, radix16_dht_avx2_gen, radix16_rdft_fwd_avx2_gen},
        {32, radix32_dht_avx2_gen, radix32_rdft_fwd_avx2_gen},
        {64, radix64_dht_avx2_gen, radix64_rdft_fwd_avx2_gen},
    };
    size_t Ks[] = {32, 128, 512, 0};
    int fails = 0;
    for (size_t i = 0; i < sizeof(cells)/sizeof(cells[0]); i++) {
        for (int ki = 0; Ks[ki] != 0; ki++) {
            fails += run_one(&cells[i], Ks[ki]);
        }
        printf("\n");
    }
    printf("%s\n", fails == 0 ? "All correctness PASS" : "Some FAIL");
    return fails;
}
