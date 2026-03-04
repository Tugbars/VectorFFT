/*
 * test_bench_r128_gen.c — Test + benchmark for generated DFT-128 N1 kernel
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>
#include <fftw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FORCE_INLINE __attribute__((always_inline)) inline
#define TARGET_AVX512 __attribute__((target("avx512f,avx512dq,fma")))
#define RESTRICT __restrict__
#define ALIGNAS_64 __attribute__((aligned(64)))

static void *r32_aligned_alloc(size_t align, size_t size) {
    void *p = NULL;
    posix_memalign(&p, align, size);
    return p;
}
static void r32_aligned_free(void *p) { free(p); }

#define aa(n) (double*)r32_aligned_alloc(64, (n)*sizeof(double))

#include "fft_radix128_avx512_n1_gen.h"

/* ── Simple driver — just the unaligned variant for now ── */
TARGET_AVX512
static void radix128_fwd_gen(size_t K,
    const double *RESTRICT ir, const double *RESTRICT ii,
    double *RESTRICT or_, double *RESTRICT oi)
{
    radix128_n1_dit_kernel_fwd_avx512(ir, ii, or_, oi, K);
}

TARGET_AVX512
static void radix128_bwd_gen(size_t K,
    const double *RESTRICT ir, const double *RESTRICT ii,
    double *RESTRICT or_, double *RESTRICT oi)
{
    radix128_n1_dit_kernel_bwd_avx512(ir, ii, or_, oi, K);
}

/* ── Naive DFT-128 for correctness ── */
static void naive_dft128(size_t K, size_t k,
    const double *in_re, const double *in_im,
    double *out_re, double *out_im, int direction)
{
    /* direction: -1=forward, +1=backward */
    for (int m = 0; m < 128; m++) {
        double sr = 0, si = 0;
        for (int n = 0; n < 128; n++) {
            double angle = direction * 2.0 * M_PI * m * n / 128.0;
            double wr = cos(angle), wi = sin(angle);
            double xr = in_re[n * K + k];
            double xi = in_im[n * K + k];
            sr += xr * wr - xi * wi;
            si += xr * wi + xi * wr;
        }
        out_re[m * K + k] = sr;
        out_im[m * K + k] = si;
    }
}

/* ── Utilities ── */
static void fill_rand(double *p, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++)
        p[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
}

static double max_abs(const double *p, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; i++) {
        double a = fabs(p[i]);
        if (a > m) m = a;
    }
    return m;
}

static double get_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* ── Correctness tests ── */

static int test_fwd_vs_naive(size_t K) {
    const size_t N = 128 * K;
    double *ir=aa(N), *ii=aa(N), *gr=aa(N), *gi=aa(N), *nr=aa(N), *ni=aa(N);
    fill_rand(ir, N, 40000+(unsigned)K);
    fill_rand(ii, N, 41000+(unsigned)K);

    radix128_fwd_gen(K, ir, ii, gr, gi);

    for (size_t k = 0; k < K; k++)
        naive_dft128(K, k, ir, ii, nr, ni, -1);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(gr[i]-nr[i]), fabs(gi[i]-ni[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(nr,N), max_abs(ni,N));
    double rel = (mag > 0) ? err / mag : err;
    int pass = (rel < 5e-13);

    printf("  fwd vs naive  K=%-5zu  rel=%.2e  %s\n", K, rel, pass?"PASS":"FAIL");
    if (!pass) printf("    abs_err=%.2e mag=%.2e\n", err, mag);
    free(ir);free(ii);free(gr);free(gi);free(nr);free(ni);
    return pass;
}

static int test_bwd_vs_naive(size_t K) {
    const size_t N = 128 * K;
    double *ir=aa(N), *ii=aa(N), *gr=aa(N), *gi=aa(N), *nr=aa(N), *ni=aa(N);
    fill_rand(ir, N, 42000+(unsigned)K);
    fill_rand(ii, N, 43000+(unsigned)K);

    radix128_bwd_gen(K, ir, ii, gr, gi);

    for (size_t k = 0; k < K; k++)
        naive_dft128(K, k, ir, ii, nr, ni, +1);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(gr[i]-nr[i]), fabs(gi[i]-ni[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(nr,N), max_abs(ni,N));
    double rel = (mag > 0) ? err / mag : err;
    int pass = (rel < 5e-13);

    printf("  bwd vs naive  K=%-5zu  rel=%.2e  %s\n", K, rel, pass?"PASS":"FAIL");
    if (!pass) printf("    abs_err=%.2e mag=%.2e\n", err, mag);
    free(ir);free(ii);free(gr);free(gi);free(nr);free(ni);
    return pass;
}

static int test_roundtrip(size_t K) {
    const size_t N = 128 * K;
    double *ir=aa(N), *ii=aa(N), *fr=aa(N), *fi=aa(N), *br=aa(N), *bi=aa(N);
    fill_rand(ir, N, 44000+(unsigned)K);
    fill_rand(ii, N, 45000+(unsigned)K);

    radix128_fwd_gen(K, ir, ii, fr, fi);
    radix128_bwd_gen(K, fr, fi, br, bi);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(br[i]-128.0*ir[i]), fabs(bi[i]-128.0*ii[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(ir,N), max_abs(ii,N));
    double rel = (mag > 0) ? err / (128.0*mag) : err;
    int pass = (rel < 1e-13);

    printf("  roundtrip     K=%-5zu  rel=%.2e  %s\n", K, rel, pass?"PASS":"FAIL");
    free(ir);free(ii);free(fr);free(fi);free(br);free(bi);
    return pass;
}

/* ── Benchmarks ── */

typedef void (*bench_fn)(size_t K,
    const double*, const double*, double*, double*);

static double bench(bench_fn fn, size_t K,
    const double *ir, const double *ii, double *or_, double *oi,
    int warmup, int trials)
{
    for (int i = 0; i < warmup; i++) fn(K, ir, ii, or_, oi);
    double best = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns();
        fn(K, ir, ii, or_, oi);
        double dt = get_ns() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

static double bench_fftw(fftw_plan p, fftw_complex *in, fftw_complex *out,
                         int warmup, int trials)
{
    for (int i = 0; i < warmup; i++) fftw_execute(p);
    double best = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns();
        fftw_execute(p);
        double dt = get_ns() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

static void run_bench(size_t K, int warmup, int trials) {
    const size_t N = 128 * K;
    double *ir=aa(N), *ii=aa(N), *or_=aa(N), *oi=aa(N);
    fill_rand(ir, N, 60000+(unsigned)K);
    fill_rand(ii, N, 61000+(unsigned)K);

    /* FFTW */
    fftw_complex *fftw_in  = fftw_alloc_complex(N);
    fftw_complex *fftw_out = fftw_alloc_complex(N);
    for (size_t k = 0; k < K; k++)
        for (int n = 0; n < 128; n++) {
            fftw_in[k*128+n][0] = ir[n*K+k];
            fftw_in[k*128+n][1] = ii[n*K+k];
        }
    int n_arr[1] = {128};
    fftw_plan plan = fftw_plan_many_dft(1, n_arr, (int)K,
        fftw_in, NULL, 1, 128, fftw_out, NULL, 1, 128,
        FFTW_FORWARD, FFTW_MEASURE);

    double ns_fftw = bench_fftw(plan, fftw_in, fftw_out, warmup, trials);
    double ns_gen = bench((bench_fn)radix128_fwd_gen, K, ir, ii, or_, oi, warmup, trials);

    printf("  K=%-6zu  FFTW=%8.0f  Gen=%8.0f  Gen/FFTW=%.2fx\n",
           K, ns_fftw, ns_gen, ns_fftw/ns_gen);

    fftw_destroy_plan(plan);
    fftw_free(fftw_in);
    fftw_free(fftw_out);
    free(ir);free(ii);free(or_);free(oi);
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Generated DFT-128 Kernel — Test & Benchmark               ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    int passed = 0, total = 0;

    printf("── Correctness: forward vs naive DFT-128 ──\n");
    { const size_t Ks[] = {8, 16, 32};
      for (int i = 0; i < 3; i++) { total++; passed += test_fwd_vs_naive(Ks[i]); } }

    printf("\n── Correctness: backward vs naive IDFT-128 ──\n");
    { const size_t Ks[] = {8, 16, 32};
      for (int i = 0; i < 3; i++) { total++; passed += test_bwd_vs_naive(Ks[i]); } }

    printf("\n── Correctness: roundtrip ──\n");
    { const size_t Ks[] = {8, 16, 64, 128};
      for (int i = 0; i < 4; i++) { total++; passed += test_roundtrip(Ks[i]); } }

    printf("\n══════════════════════════════════════════\n");
    printf("  %d/%d passed  %s\n", passed, total,
           passed==total ? "✓ ALL PASSED" : "✗ FAILURES");
    printf("══════════════════════════════════════════\n");

    if (passed != total) return 1;

    printf("\n── Benchmark: Gen DFT-128 vs FFTW (ns, forward) ──\n");
    printf("  %-8s  %-8s  %-8s  %-9s\n", "K", "FFTW", "Gen", "Gen/FFTW");

    run_bench(8,    500, 2000);
    run_bench(16,   500, 2000);
    run_bench(32,   200, 1000);
    run_bench(64,   200, 1000);
    run_bench(128,  100, 500);
    run_bench(256,  100, 500);
    run_bench(512,  50,  200);
    run_bench(1024, 50,  200);
    run_bench(2048, 20,  200);

    fftw_cleanup();
    return 0;
}
