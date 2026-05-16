/* bench_isa.c — Validate and bench AVX-512 vs AVX2 generated codelets.
 *
 * Strategy:
 *   1. Run AVX-512 codelet on input → result_512
 *   2. Run AVX2 codelet on same input → result_256
 *   3. Compare. They should be bit-equal at the level of FP rounding
 *      (both use the same FMA, just at different lane widths).
 *
 * We do not have a hand-coded AVX2 codelet from the user's Python
 * generator on hand for direct A/B; this bench validates that our
 * AVX2 path produces the same numerical result as our AVX-512 path
 * and times the difference.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

/* Forward declarations of the codelets. */
__attribute__((target("avx512f")))
void radix16_t1_dit_fwd_avx512(
    double * __restrict__, double * __restrict__,
    const double * __restrict__, const double * __restrict__,
    size_t, size_t);

__attribute__((target("avx2,fma")))
void radix16_t1_dit_fwd_avx2(
    double * __restrict__, double * __restrict__,
    const double * __restrict__, const double * __restrict__,
    size_t, size_t);

__attribute__((target("avx512f")))
void radix8_t1_dit_fwd_avx512(
    double * __restrict__, double * __restrict__,
    const double * __restrict__, const double * __restrict__,
    size_t, size_t);

__attribute__((target("avx2,fma")))
void radix8_t1_dit_fwd_avx2(
    double * __restrict__, double * __restrict__,
    const double * __restrict__, const double * __restrict__,
    size_t, size_t);

static double *aa(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc fail\n"); exit(1);
    }
    return (double *)p;
}

static void fr(double *p, size_t n, unsigned s) {
    for (size_t i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        p[i] = (double)((int)(s >> 8) & 0x7fffff) / (double)0x800000 - 0.5;
    }
}

static double max_rel(const double *a, const double *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        double s = fabs(a[i]) + fabs(b[i]) + 1e-30;
        double r = d / s;
        if (r > m) m = r;
    }
    return m;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

static double bn(void (*f)(void), int repeat, int trials) {
    double best = 1e18;
    for (int i = 0; i < 100; i++) f();
    for (int t = 0; t < trials; t++) {
        double t0 = now();
        for (int i = 0; i < repeat; i++) f();
        double dt = (now() - t0) / (double)repeat;
        if (dt < best) best = dt;
    }
    return best;
}

static size_t g_K;
static int g_R;
static double *g_rio_re_512, *g_rio_im_512;
static double *g_rio_re_256, *g_rio_im_256;
static double *g_tw_re, *g_tw_im;
static double *g_in_re, *g_in_im;

static void call_512(void) {
    if (g_R == 16)
        radix16_t1_dit_fwd_avx512(g_rio_re_512, g_rio_im_512, g_tw_re, g_tw_im, g_K, g_K);
    else
        radix8_t1_dit_fwd_avx512(g_rio_re_512, g_rio_im_512, g_tw_re, g_tw_im, g_K, g_K);
}
static void call_256(void) {
    if (g_R == 16)
        radix16_t1_dit_fwd_avx2(g_rio_re_256, g_rio_im_256, g_tw_re, g_tw_im, g_K, g_K);
    else
        radix8_t1_dit_fwd_avx2(g_rio_re_256, g_rio_im_256, g_tw_re, g_tw_im, g_K, g_K);
}

int main(int argc, char **argv) {
    g_R = (argc > 1) ? atoi(argv[1]) : 16;
    g_K = (argc > 2) ? (size_t)atoi(argv[2]) : 1024;
    if (g_K < 8 || g_K % 8 != 0) {
        fprintf(stderr, "K must be multiple of 8 (LCM of vector widths)\n"); return 1;
    }
    if (g_R != 8 && g_R != 16) {
        fprintf(stderr, "R must be 8 or 16\n"); return 1;
    }

    int legs = g_R;
    int tw_legs = g_R - 1;
    g_in_re = aa(legs * g_K);
    g_in_im = aa(legs * g_K);
    g_rio_re_512 = aa(legs * g_K);
    g_rio_im_512 = aa(legs * g_K);
    g_rio_re_256 = aa(legs * g_K);
    g_rio_im_256 = aa(legs * g_K);
    g_tw_re = aa(tw_legs * g_K);
    g_tw_im = aa(tw_legs * g_K);

    fr(g_in_re, legs * g_K, 0xa1);
    fr(g_in_im, legs * g_K, 0xa2);
    fr(g_tw_re, tw_legs * g_K, 0xb1);
    fr(g_tw_im, tw_legs * g_K, 0xb2);

    memcpy(g_rio_re_512, g_in_re, legs * g_K * sizeof(double));
    memcpy(g_rio_im_512, g_in_im, legs * g_K * sizeof(double));
    memcpy(g_rio_re_256, g_in_re, legs * g_K * sizeof(double));
    memcpy(g_rio_im_256, g_in_im, legs * g_K * sizeof(double));

    call_512();
    call_256();

    double err_re = max_rel(g_rio_re_512, g_rio_re_256, legs * g_K);
    double err_im = max_rel(g_rio_im_512, g_rio_im_256, legs * g_K);
    double err = err_re > err_im ? err_re : err_im;
    if (err > 1e-10) {
        printf("CORRECTNESS FAIL R=%d K=%zu: re=%.2e im=%.2e\n", g_R, g_K, err_re, err_im);
        return 2;
    }

    int repeat = 5000, trials = 7;
    double t512 = bn(call_512, repeat, trials);
    double t256 = bn(call_256, repeat, trials);
    /* Throughput equivalence: AVX-512 processes 8 lanes/iter, AVX2 4.
     * For a fair comparison at equal "scalar work", AVX2 needs to
     * process 2x the iterations to match. Or equivalently, scale t256/2.
     *
     * Per iteration of the inner loop:
     *   AVX-512: 8 doubles × R legs = 8R scalar elements processed
     *   AVX2:    4 doubles × R legs = 4R scalar elements processed
     * So if t512 = X ns/iter doing 8R scalars, throughput = 8R/X scalars/ns
     *    if t256 = Y ns/iter doing 4R scalars, throughput = 4R/Y scalars/ns
     * Ratio of throughputs (AVX-512 / AVX2) = (8R/X) / (4R/Y) = 2Y/X = 2*(t256/t512)
     * If 2Y/X > 1, AVX-512 is faster per scalar element. */
    double thr_ratio = 2.0 * t256 / t512;
    printf("R=%d K=%5zu  AVX512=%7.1fns  AVX2=%7.1fns  AVX2/AVX512=%.3f  thr_ratio_512_to_256=%.3f\n",
           g_R, g_K, t512, t256, t256 / t512, thr_ratio);
    return 0;
}
