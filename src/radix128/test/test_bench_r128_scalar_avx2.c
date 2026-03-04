/*
 * test_bench_r128_scalar_avx2.c — Test + benchmark for generated DFT-128
 * Covers both scalar and AVX2 backends.
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

#define RESTRICT __restrict__

static void *aligned_alloc_64(size_t size) {
    void *p = NULL;
    posix_memalign(&p, 64, size);
    return p;
}
#define aa(n) (double*)aligned_alloc_64((n)*sizeof(double))

/* Include generated kernels */
#include "fft_radix128_scalar_n1_gen.h"
#include "fft_radix128_avx2_n1_gen.h"

/* ── Naive DFT-128 ── */
static void naive_dft128(int direction, size_t K, size_t k,
    const double *in_re, const double *in_im,
    double *out_re, double *out_im)
{
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

/* ── Utils ── */
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

static int test_fwd(const char *label,
    void (*fn)(const double*, const double*, double*, double*, size_t),
    size_t K)
{
    const size_t N = 128 * K;
    double *ir=aa(N), *ii=aa(N), *gr=aa(N), *gi=aa(N), *nr=aa(N), *ni=aa(N);
    fill_rand(ir, N, 10000+(unsigned)K);
    fill_rand(ii, N, 20000+(unsigned)K);

    fn(ir, ii, gr, gi, K);
    for (size_t k = 0; k < K; k++)
        naive_dft128(-1, K, k, ir, ii, nr, ni);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(gr[i]-nr[i]), fabs(gi[i]-ni[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(nr,N), max_abs(ni,N));
    double rel = (mag > 0) ? err / mag : err;
    int pass = (rel < 5e-13);

    printf("  %-8s fwd K=%-4zu  rel=%.2e  %s\n", label, K, rel, pass?"PASS":"FAIL");
    free(ir);free(ii);free(gr);free(gi);free(nr);free(ni);
    return pass;
}

static int test_bwd(const char *label,
    void (*fn)(const double*, const double*, double*, double*, size_t),
    size_t K)
{
    const size_t N = 128 * K;
    double *ir=aa(N), *ii=aa(N), *gr=aa(N), *gi=aa(N), *nr=aa(N), *ni=aa(N);
    fill_rand(ir, N, 30000+(unsigned)K);
    fill_rand(ii, N, 40000+(unsigned)K);

    fn(ir, ii, gr, gi, K);
    for (size_t k = 0; k < K; k++)
        naive_dft128(+1, K, k, ir, ii, nr, ni);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(gr[i]-nr[i]), fabs(gi[i]-ni[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(nr,N), max_abs(ni,N));
    double rel = (mag > 0) ? err / mag : err;
    int pass = (rel < 5e-13);

    printf("  %-8s bwd K=%-4zu  rel=%.2e  %s\n", label, K, rel, pass?"PASS":"FAIL");
    free(ir);free(ii);free(gr);free(gi);free(nr);free(ni);
    return pass;
}

static int test_roundtrip(const char *label,
    void (*fwd_fn)(const double*, const double*, double*, double*, size_t),
    void (*bwd_fn)(const double*, const double*, double*, double*, size_t),
    size_t K)
{
    const size_t N = 128 * K;
    double *ir=aa(N), *ii=aa(N), *fr=aa(N), *fi=aa(N), *br=aa(N), *bi=aa(N);
    fill_rand(ir, N, 50000+(unsigned)K);
    fill_rand(ii, N, 60000+(unsigned)K);

    fwd_fn(ir, ii, fr, fi, K);
    bwd_fn(fr, fi, br, bi, K);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(br[i]-128.0*ir[i]), fabs(bi[i]-128.0*ii[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(ir,N), max_abs(ii,N));
    double rel = (mag > 0) ? err / (128.0*mag) : err;
    int pass = (rel < 1e-13);

    printf("  %-8s rt  K=%-4zu  rel=%.2e  %s\n", label, K, rel, pass?"PASS":"FAIL");
    free(ir);free(ii);free(fr);free(fi);free(br);free(bi);
    return pass;
}

/* ── Cross-ISA: scalar vs avx2 bit-exact comparison ── */

static int test_cross_isa(size_t K) {
    const size_t N = 128 * K;
    double *ir=aa(N), *ii=aa(N);
    double *sr=aa(N), *si=aa(N), *ar=aa(N), *ai=aa(N);
    fill_rand(ir, N, 70000+(unsigned)K);
    fill_rand(ii, N, 71000+(unsigned)K);

    radix128_n1_dit_kernel_fwd_scalar(ir, ii, sr, si, K);
    radix128_n1_dit_kernel_fwd_avx2(ir, ii, ar, ai, K);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(sr[i]-ar[i]), fabs(si[i]-ai[i]));
        if (e > err) err = e;
    }
    int pass = (err == 0.0);  /* should be bit-exact */

    printf("  cross    fwd K=%-4zu  maxdiff=%.2e  %s\n", K, err,
           pass ? "BIT-EXACT" : (err < 1e-14 ? "PASS (near)" : "FAIL"));
    free(ir);free(ii);free(sr);free(si);free(ar);free(ai);
    return pass || (err < 1e-14);
}

/* ── Wrappers for uniform calling convention ── */

static void wrap_scalar_fwd(const double *ir, const double *ii,
                            double *or_, double *oi, size_t K) {
    radix128_n1_dit_kernel_fwd_scalar(ir, ii, or_, oi, K);
}
static void wrap_scalar_bwd(const double *ir, const double *ii,
                            double *or_, double *oi, size_t K) {
    radix128_n1_dit_kernel_bwd_scalar(ir, ii, or_, oi, K);
}
static void wrap_avx2_fwd(const double *ir, const double *ii,
                          double *or_, double *oi, size_t K) {
    radix128_n1_dit_kernel_fwd_avx2(ir, ii, or_, oi, K);
}
static void wrap_avx2_bwd(const double *ir, const double *ii,
                          double *or_, double *oi, size_t K) {
    radix128_n1_dit_kernel_bwd_avx2(ir, ii, or_, oi, K);
}

/* ── Benchmarks ── */

typedef void (*kern_fn)(const double*, const double*, double*, double*, size_t);

static double bench_kern(kern_fn fn, size_t K,
    const double *ir, const double *ii, double *or_, double *oi,
    int warmup, int trials)
{
    for (int i = 0; i < warmup; i++) fn(ir, ii, or_, oi, K);
    double best = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns();
        fn(ir, ii, or_, oi, K);
        double dt = get_ns() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

static void run_bench(size_t K, int warmup, int trials) {
    const size_t N = 128 * K;
    double *ir=aa(N), *ii=aa(N), *or_=aa(N), *oi=aa(N);
    fill_rand(ir, N, 80000+(unsigned)K);
    fill_rand(ii, N, 90000+(unsigned)K);

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

    for (int i = 0; i < warmup; i++) fftw_execute(plan);
    double best_fftw = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns();
        fftw_execute(plan);
        double dt = get_ns() - t0;
        if (dt < best_fftw) best_fftw = dt;
    }

    double ns_scalar = bench_kern(wrap_scalar_fwd, K, ir, ii, or_, oi, warmup, trials);
    double ns_avx2   = bench_kern(wrap_avx2_fwd,   K, ir, ii, or_, oi, warmup, trials);

    printf("  K=%-5zu  FFTW=%8.0f  Scalar=%8.0f  AVX2=%8.0f  "
           "S/FFTW=%.2fx  A/FFTW=%.2fx  A/S=%.1fx\n",
           K, best_fftw, ns_scalar, ns_avx2,
           best_fftw/ns_scalar, best_fftw/ns_avx2, ns_scalar/ns_avx2);

    fftw_destroy_plan(plan);
    fftw_free(fftw_in);
    fftw_free(fftw_out);
    free(ir);free(ii);free(or_);free(oi);
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  DFT-128 N1 — Scalar + AVX2 Generated Kernels             ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    int passed = 0, total = 0;

    /* ── Scalar correctness ── */
    printf("── Scalar forward ──\n");
    { size_t Ks[] = {1, 2, 3, 7, 8, 16};
      for (int i = 0; i < 6; i++) { total++; passed += test_fwd("scalar", wrap_scalar_fwd, Ks[i]); } }

    printf("\n── Scalar backward ──\n");
    { size_t Ks[] = {1, 2, 8};
      for (int i = 0; i < 3; i++) { total++; passed += test_bwd("scalar", wrap_scalar_bwd, Ks[i]); } }

    printf("\n── Scalar roundtrip ──\n");
    { size_t Ks[] = {1, 4, 8, 16};
      for (int i = 0; i < 4; i++) { total++; passed += test_roundtrip("scalar", wrap_scalar_fwd, wrap_scalar_bwd, Ks[i]); } }

    /* ── AVX2 correctness ── */
    printf("\n── AVX2 forward ──\n");
    { size_t Ks[] = {4, 8, 16, 32};
      for (int i = 0; i < 4; i++) { total++; passed += test_fwd("avx2", wrap_avx2_fwd, Ks[i]); } }

    printf("\n── AVX2 backward ──\n");
    { size_t Ks[] = {4, 8, 16};
      for (int i = 0; i < 3; i++) { total++; passed += test_bwd("avx2", wrap_avx2_bwd, Ks[i]); } }

    printf("\n── AVX2 roundtrip ──\n");
    { size_t Ks[] = {4, 8, 32, 64};
      for (int i = 0; i < 4; i++) { total++; passed += test_roundtrip("avx2", wrap_avx2_fwd, wrap_avx2_bwd, Ks[i]); } }

    /* ── Cross-ISA ── */
    printf("\n── Cross-ISA (scalar vs AVX2) ──\n");
    { size_t Ks[] = {4, 8, 16};
      for (int i = 0; i < 3; i++) { total++; passed += test_cross_isa(Ks[i]); } }

    printf("\n══════════════════════════════════════════\n");
    printf("  %d/%d passed  %s\n", passed, total,
           passed==total ? "✓ ALL PASSED" : "✗ FAILURES");
    printf("══════════════════════════════════════════\n");

    if (passed != total) return 1;

    /* ── Benchmarks ── */
    printf("\n── Benchmark: Scalar vs AVX2 vs FFTW (ns, fwd) ──\n");
    printf("  %-7s  %-8s  %-8s  %-8s  %-7s  %-7s  %s\n",
           "K", "FFTW", "Scalar", "AVX2", "S/FFTW", "A/FFTW", "A/S");

    run_bench(1,    500, 3000);
    run_bench(4,    500, 3000);
    run_bench(8,    500, 2000);
    run_bench(16,   500, 2000);
    run_bench(32,   200, 1000);
    run_bench(64,   200, 1000);
    run_bench(128,  100, 500);
    run_bench(256,  100, 500);
    run_bench(512,  50,  200);
    run_bench(1024, 50,  200);

    fftw_cleanup();
    return 0;
}
