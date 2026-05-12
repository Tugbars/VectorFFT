/* bench_dct4_3pass.c — DCT-IV: fused DAG codelet vs production-style 3-pass.
 *
 * Production runtime (per src/core/dct4.h):
 *   Phase 1: build z[m] = x[2m] - i·x[N-1-2m], apply pre-twiddle exp(iπm/N)
 *            → psi[m]   (one O(N·K) memory pass)
 *   Phase 2: N/2-point c2c IFFT (backward) of psi → ifft   (codelet call)
 *   Phase 3: post-twiddle 2·exp(iπ(4k'+1)/(4N))·ifft → Z,
 *            then Y[2k']=Re(Z), Y[N-1-2k']=Im(Z)   (another O(N·K) pass)
 *
 * Our fused DAG codelet collapses all 3 phases into a single straight-line
 * codelet — no inter-phase memory traffic.
 *
 * Same expected shape as DST-II/III vs 3-pass: fused wins big at small K.
 */
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

__attribute__((target("avx2,fma"))) void radix8_dct4_avx2_gen(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
__attribute__((target("avx2,fma"))) void radix16_dct4_avx2_gen(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
__attribute__((target("avx2,fma"))) void radix32_dct4_avx2_gen(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
__attribute__((target("avx2,fma"))) void radix64_dct4_avx2_gen(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);

/* c2c backward (IFFT) for inner kernel at N/2. */
__attribute__((target("avx2,fma"))) void radix4_n1_bwd_avx2_gen(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
__attribute__((target("avx2,fma"))) void radix8_n1_bwd_avx2_gen(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
__attribute__((target("avx2,fma"))) void radix16_n1_bwd_avx2_gen(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
__attribute__((target("avx2,fma"))) void radix32_n1_bwd_avx2_gen(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);

typedef void (*dct4_fn_t)(const double *, const double *, double *, double *,
                          const double *, const double *, size_t);
typedef void (*c2c_bwd_fn_t)(const double *, const double *, double *, double *,
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

/* Phase 1: pre-twiddle. psi[m*K+b] = (x[2m]-i·x[N-1-2m]) · exp(iπm/N). */
__attribute__((target("avx2,fma")))
static void pre_twiddle(const double *x, double *psi_re, double *psi_im,
                        const double *pre_c, const double *pre_s,
                        int N, size_t K) {
    int half = N / 2;
    for (int m = 0; m < half; m++) {
        __m256d c = _mm256_set1_pd(pre_c[m]);
        __m256d s = _mm256_set1_pd(pre_s[m]);
        const double *xa = x + (size_t)(2 * m) * K;
        const double *xb = x + (size_t)(N - 1 - 2 * m) * K;
        double *pr = psi_re + (size_t)m * K;
        double *pi = psi_im + (size_t)m * K;
        for (size_t b = 0; b < K; b += 4) {
            __m256d zr = _mm256_loadu_pd(xa + b);
            __m256d zi = _mm256_xor_pd(_mm256_loadu_pd(xb + b),
                                       _mm256_set1_pd(-0.0));
            /* (zr + i·zi) · (c + i·s) = (zr·c - zi·s) + i(zr·s + zi·c) */
            __m256d pr_v = _mm256_fmsub_pd(zr, c, _mm256_mul_pd(zi, s));
            __m256d pi_v = _mm256_fmadd_pd(zr, s, _mm256_mul_pd(zi, c));
            _mm256_storeu_pd(pr + b, pr_v);
            _mm256_storeu_pd(pi + b, pi_v);
        }
    }
}

/* Phase 3: post-twiddle + extract.
 * Z[k'] = (2c + 2i·s)·(ifft_re + i·ifft_im); Y[2k']=Re(Z), Y[N-1-2k']=Im(Z). */
__attribute__((target("avx2,fma")))
static void post_twiddle(const double *ifft_re, const double *ifft_im,
                         double *y, const double *post_2c, const double *post_2s,
                         int N, size_t K) {
    int half = N / 2;
    for (int kp = 0; kp < half; kp++) {
        __m256d c = _mm256_set1_pd(post_2c[kp]);
        __m256d s = _mm256_set1_pd(post_2s[kp]);
        const double *ir = ifft_re + (size_t)kp * K;
        const double *ii = ifft_im + (size_t)kp * K;
        double *y_even = y + (size_t)(2 * kp)         * K;
        double *y_odd  = y + (size_t)(N - 1 - 2 * kp) * K;
        for (size_t b = 0; b < K; b += 4) {
            __m256d xr = _mm256_loadu_pd(ir + b);
            __m256d xi = _mm256_loadu_pd(ii + b);
            __m256d zr = _mm256_fmsub_pd(c, xr, _mm256_mul_pd(s, xi));
            __m256d zi = _mm256_fmadd_pd(c, xi, _mm256_mul_pd(s, xr));
            _mm256_storeu_pd(y_even + b, zr);
            _mm256_storeu_pd(y_odd  + b, zi);
        }
    }
}

/* Per-N dispatcher state */
typedef struct {
    int N;
    dct4_fn_t fused;
    c2c_bwd_fn_t ifft;
    double *pre_c, *pre_s, *post_2c, *post_2s;
} cell_t;

static void init_twiddles(cell_t *c) {
    int N = c->N, half = N / 2;
    c->pre_c   = aa(half);
    c->pre_s   = aa(half);
    c->post_2c = aa(half);
    c->post_2s = aa(half);
    for (int m = 0; m < half; m++) {
        double phi = M_PI * m / N;
        c->pre_c[m] = cos(phi);
        c->pre_s[m] = sin(phi);
    }
    for (int kp = 0; kp < half; kp++) {
        double phi = M_PI * (4 * kp + 1) / (4.0 * N);
        c->post_2c[kp] = 2.0 * cos(phi);
        c->post_2s[kp] = 2.0 * sin(phi);
    }
}

static size_t g_K;
static cell_t *g_cell;
static double *g_in, *g_out_3pass, *g_out_fused, *g_out_im;
static double *g_psi_re, *g_psi_im, *g_ifft_re, *g_ifft_im, *g_dummy;

static void call_3pass(void) {
    pre_twiddle(g_in, g_psi_re, g_psi_im,
                g_cell->pre_c, g_cell->pre_s, g_cell->N, g_K);
    g_cell->ifft(g_psi_re, g_psi_im, g_ifft_re, g_ifft_im,
                 g_dummy, g_dummy, g_K);
    post_twiddle(g_ifft_re, g_ifft_im, g_out_3pass,
                 g_cell->post_2c, g_cell->post_2s, g_cell->N, g_K);
}
static void call_fused(void) {
    g_cell->fused(g_in, g_dummy, g_out_fused, g_out_im,
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
    g_in         = aa((size_t)N * K);
    g_out_3pass  = aa((size_t)N * K);
    g_out_fused  = aa((size_t)N * K);
    g_out_im     = aa((size_t)N * K);
    g_psi_re     = aa((size_t)(N/2) * K);
    g_psi_im     = aa((size_t)(N/2) * K);
    g_ifft_re    = aa((size_t)(N/2) * K);
    g_ifft_im    = aa((size_t)(N/2) * K);
    g_dummy      = aa((size_t)N * K);
    fr(g_in, (size_t)N * K, 0xC4u + (unsigned)N);
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
        printf("N=%-3d K=%-5zu  FAIL  3pass vs fused diff=%.2e\n", N, K, err);
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
    printf("  DCT-IV: OCaml fused DAG codelet vs production-style 3-pass\n");
    printf("  (pre-twiddle + c2c-N/2 IFFT + post-twiddle)\n");
    printf("================================================================\n");
    cell_t cells[] = {
        {8,  radix8_dct4_avx2_gen,  radix4_n1_bwd_avx2_gen, NULL,NULL,NULL,NULL},
        {16, radix16_dct4_avx2_gen, radix8_n1_bwd_avx2_gen, NULL,NULL,NULL,NULL},
        {32, radix32_dct4_avx2_gen, radix16_n1_bwd_avx2_gen,NULL,NULL,NULL,NULL},
        {64, radix64_dct4_avx2_gen, radix32_n1_bwd_avx2_gen,NULL,NULL,NULL,NULL},
    };
    size_t Ks[] = {32, 64, 128, 256, 512, 1024, 0};
    int fails = 0;
    for (size_t i = 0; i < sizeof(cells)/sizeof(cells[0]); i++) {
        init_twiddles(&cells[i]);
        for (int ki = 0; Ks[ki] != 0; ki++) {
            fails += run_one(&cells[i], Ks[ki]);
        }
        printf("\n");
    }
    printf("%s\n", fails == 0 ? "All correctness PASS" : "Some FAIL");
    return fails;
}
