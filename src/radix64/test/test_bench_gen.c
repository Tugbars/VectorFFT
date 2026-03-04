/**
 * @file test_bench_gen.c
 * @brief Test & benchmark: generated DFT-64 kernel vs Architecture A vs FFTW
 *
 * Build:
 *   gcc -O2 -mavx512f -mavx512dq -mfma -Wno-unused-function \
 *       test_bench_gen.c -o test_bench_gen -lfftw3 -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

#include "fft_radix32_platform.h"
#include "fft_radix8_avx512.h"
#include "fft_radix64_avx512_n1.h"
#include "fft_radix64_avx512_n1_gen_driver.h"

/*==========================================================================*/

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static double *aa(size_t n) {
    double *p = (double *)r32_aligned_alloc(64, n * sizeof(double));
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    memset(p, 0, n * sizeof(double));
    return p;
}

static void fill_rand(double *buf, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        buf[i] = ((double)(seed >> 16) / 32768.0) - 1.0;
    }
}

static double max_abs(const double *a, size_t n) {
    double mx = 0;
    for (size_t i = 0; i < n; i++) {
        double v = fabs(a[i]);
        if (v > mx) mx = v;
    }
    return mx;
}

/* Naive DFT-64 for verification */
static void naive_dft64(size_t K, size_t k,
                        const double *in_re, const double *in_im,
                        double *out_re, double *out_im)
{
    for (int m = 0; m < 64; m++) {
        double sr = 0, si = 0;
        for (int n = 0; n < 64; n++) {
            double angle = -2.0 * M_PI * m * n / 64.0;
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

/*==========================================================================
 * TESTS
 *=========================================================================*/

static int test_gen_vs_naive(size_t K) {
    const size_t N = 64 * K;
    double *ir=aa(N),*ii=aa(N),*gr=aa(N),*gi=aa(N),*nr=aa(N),*ni=aa(N);
    fill_rand(ir,N,50000+(unsigned)K);
    fill_rand(ii,N,51000+(unsigned)K);

    /* Generated kernel */
    radix64_n1_forward_avx512_gen(K, ir, ii, gr, gi);

    /* Naive */
    for (size_t k = 0; k < K; k++)
        naive_dft64(K, k, ir, ii, nr, ni);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(gr[i]-nr[i]), fabs(gi[i]-ni[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(nr,N), max_abs(ni,N));
    double rel = (mag > 0) ? err / mag : err;
    int pass = (rel < 5e-14);

    printf("  gen vs naive  K=%-5zu  rel=%.2e  %s\n", K, rel, pass?"PASS":"FAIL");
    r32_aligned_free(ir);r32_aligned_free(ii);r32_aligned_free(gr);
    r32_aligned_free(gi);r32_aligned_free(nr);r32_aligned_free(ni);
    return pass;
}

static int test_gen_vs_archA(size_t K) {
    const size_t N = 64 * K;
    double *ir=aa(N),*ii=aa(N);
    double *ar=aa(N),*ai=aa(N);  /* Arch A */
    double *gr=aa(N),*gi=aa(N);  /* Gen */
    fill_rand(ir,N,52000+(unsigned)K);
    fill_rand(ii,N,53000+(unsigned)K);

    radix64_n1_forward_avx512(K, ir, ii, ar, ai);
    radix64_n1_forward_avx512_gen(K, ir, ii, gr, gi);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(gr[i]-ar[i]), fabs(gi[i]-ai[i]));
        if (e > err) err = e;
    }
    int pass = (err < 1e-13);  /* Different decomposition order → few ULP diffs */

    printf("  gen vs archA  K=%-5zu  err=%.2e  %s\n", K, err, pass?"PASS":"FAIL");
    r32_aligned_free(ir);r32_aligned_free(ii);r32_aligned_free(ar);
    r32_aligned_free(ai);r32_aligned_free(gr);r32_aligned_free(gi);
    return pass;
}

static int test_gen_roundtrip(size_t K) {
    const size_t N = 64 * K;
    double *ir=aa(N),*ii=aa(N),*fr=aa(N),*fi=aa(N),*br=aa(N),*bi=aa(N);
    fill_rand(ir,N,54000+(unsigned)K);
    fill_rand(ii,N,55000+(unsigned)K);

    radix64_n1_forward_avx512_gen(K, ir, ii, fr, fi);
    radix64_n1_backward_avx512_gen(K, fr, fi, br, bi);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(br[i]-64.0*ir[i]), fabs(bi[i]-64.0*ii[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(ir,N), max_abs(ii,N));
    double rel = (mag > 0) ? err / (64.0*mag) : err;
    int pass = (rel < 1e-13);

    printf("  gen roundtrip K=%-5zu  rel=%.2e  %s\n", K, rel, pass?"PASS":"FAIL");
    r32_aligned_free(ir);r32_aligned_free(ii);r32_aligned_free(fr);
    r32_aligned_free(fi);r32_aligned_free(br);r32_aligned_free(bi);
    return pass;
}

/* Naive backward DFT-64 (IDFT without 1/N) */
static void naive_idft64(size_t K, size_t k,
                         const double *in_re, const double *in_im,
                         double *out_re, double *out_im)
{
    for (int m = 0; m < 64; m++) {
        double sr = 0, si = 0;
        for (int n = 0; n < 64; n++) {
            double angle = +2.0 * M_PI * m * n / 64.0; /* +sign for backward */
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

static int test_gen_bwd_vs_naive(size_t K) {
    const size_t N = 64 * K;
    double *ir=aa(N),*ii=aa(N),*gr=aa(N),*gi=aa(N),*nr=aa(N),*ni=aa(N);
    fill_rand(ir,N,56000+(unsigned)K);
    fill_rand(ii,N,57000+(unsigned)K);

    radix64_n1_backward_avx512_gen(K, ir, ii, gr, gi);

    for (size_t k = 0; k < K; k++)
        naive_idft64(K, k, ir, ii, nr, ni);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(gr[i]-nr[i]), fabs(gi[i]-ni[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(nr,N), max_abs(ni,N));
    double rel = (mag > 0) ? err / mag : err;
    int pass = (rel < 5e-14);

    printf("  bwd vs naive  K=%-5zu  rel=%.2e  %s\n", K, rel, pass?"PASS":"FAIL");
    r32_aligned_free(ir);r32_aligned_free(ii);r32_aligned_free(gr);
    r32_aligned_free(gi);r32_aligned_free(nr);r32_aligned_free(ni);
    return pass;
}

/*==========================================================================
 * BENCHMARK
 *=========================================================================*/

typedef void (*fwd_fn_t)(size_t K,
    const double *ir, const double *ii,
    double *or_, double *oi);

static double bench(fwd_fn_t fn, size_t K,
                    const double *ir, const double *ii,
                    double *or_, double *oi,
                    int warmup, int trials)
{
    for (int i = 0; i < warmup; i++)
        fn(K, ir, ii, or_, oi);

    double *times = (double *)malloc(trials * sizeof(double));
    for (int t = 0; t < trials; t++) {
        double t0 = now_ns();
        fn(K, ir, ii, or_, oi);
        double t1 = now_ns();
        times[t] = t1 - t0;
    }

    for (int i = 0; i < trials - 1; i++)
        for (int j = i + 1; j < trials; j++)
            if (times[j] < times[i]) {
                double tmp = times[i]; times[i] = times[j]; times[j] = tmp;
            }
    double median = times[trials / 2];
    free(times);
    return median;
}

static void wrap_archA(size_t K,
    const double *ir, const double *ii, double *or_, double *oi)
{
    radix64_n1_forward_avx512(K, ir, ii, or_, oi);
}

static void wrap_gen(size_t K,
    const double *ir, const double *ii, double *or_, double *oi)
{
    radix64_n1_forward_avx512_gen(K, ir, ii, or_, oi);
}

static double bench_fftw(fftw_plan plan,
                         fftw_complex *in, fftw_complex *out,
                         int warmup, int trials)
{
    for (int i = 0; i < warmup; i++)
        fftw_execute_dft(plan, in, out);

    double *times = (double *)malloc(trials * sizeof(double));
    for (int t = 0; t < trials; t++) {
        double t0 = now_ns();
        fftw_execute_dft(plan, in, out);
        double t1 = now_ns();
        times[t] = t1 - t0;
    }

    for (int i = 0; i < trials - 1; i++)
        for (int j = i + 1; j < trials; j++)
            if (times[j] < times[i]) {
                double tmp = times[i]; times[i] = times[j]; times[j] = tmp;
            }
    double median = times[trials / 2];
    free(times);
    return median;
}

static void run_bench(size_t K, int warmup, int trials)
{
    const size_t N = 64 * K;
    double *ir=aa(N),*ii=aa(N),*or_=aa(N),*oi=aa(N);
    fill_rand(ir,N,60000+(unsigned)K);
    fill_rand(ii,N,61000+(unsigned)K);

    /* FFTW */
    fftw_complex *fftw_in  = fftw_alloc_complex(N);
    fftw_complex *fftw_out = fftw_alloc_complex(N);
    for (size_t k = 0; k < K; k++)
        for (int n = 0; n < 64; n++) {
            fftw_in[k*64+n][0] = ir[n*K+k];
            fftw_in[k*64+n][1] = ii[n*K+k];
        }
    int n_arr[1] = {64};
    fftw_plan plan = fftw_plan_many_dft(1, n_arr, (int)K,
        fftw_in, NULL, 1, 64, fftw_out, NULL, 1, 64,
        FFTW_FORWARD, FFTW_MEASURE);

    double ns_fftw = bench_fftw(plan, fftw_in, fftw_out, warmup, trials);
    double ns_archA = bench(wrap_archA, K, ir, ii, or_, oi, warmup, trials);
    double ns_gen   = bench(wrap_gen,   K, ir, ii, or_, oi, warmup, trials);

    double speedup_vs_A   = ns_archA / ns_gen;
    double speedup_vs_fftw = ns_fftw / ns_gen;

    printf("  K=%-6zu  FFTW=%8.0f  ArchA=%8.0f  Gen=%8.0f  "
           "Gen/A=%.2fx  Gen/FFTW=%.2fx\n",
           K, ns_fftw, ns_archA, ns_gen, speedup_vs_A, speedup_vs_fftw);

    fftw_destroy_plan(plan);
    fftw_free(fftw_in);
    fftw_free(fftw_out);
    r32_aligned_free(ir);r32_aligned_free(ii);
    r32_aligned_free(or_);r32_aligned_free(oi);
}

/* ── Backward-specific benchmarks ── */

static void wrap_archA_bwd(size_t K,
    const double *ir, const double *ii, double *or_, double *oi)
{
    radix64_n1_backward_avx512(K, ir, ii, or_, oi);
}

static void wrap_gen_bwd(size_t K,
    const double *ir, const double *ii, double *or_, double *oi)
{
    radix64_n1_backward_avx512_gen(K, ir, ii, or_, oi);
}

static void run_bench_bwd(size_t K, int warmup, int trials)
{
    const size_t N = 64 * K;
    double *ir=aa(N),*ii=aa(N),*or_=aa(N),*oi=aa(N);
    fill_rand(ir,N,70000+(unsigned)K);
    fill_rand(ii,N,71000+(unsigned)K);

    /* FFTW backward */
    fftw_complex *fftw_in  = fftw_alloc_complex(N);
    fftw_complex *fftw_out = fftw_alloc_complex(N);
    for (size_t k = 0; k < K; k++)
        for (int n = 0; n < 64; n++) {
            fftw_in[k*64+n][0] = ir[n*K+k];
            fftw_in[k*64+n][1] = ii[n*K+k];
        }
    int n_arr[1] = {64};
    fftw_plan plan = fftw_plan_many_dft(1, n_arr, (int)K,
        fftw_in, NULL, 1, 64, fftw_out, NULL, 1, 64,
        FFTW_BACKWARD, FFTW_MEASURE);

    double ns_fftw  = bench_fftw(plan, fftw_in, fftw_out, warmup, trials);
    double ns_archA = bench(wrap_archA_bwd, K, ir, ii, or_, oi, warmup, trials);
    double ns_gen   = bench(wrap_gen_bwd,   K, ir, ii, or_, oi, warmup, trials);

    printf("  K=%-6zu  FFTW=%8.0f  ArchA=%8.0f  Gen=%8.0f  "
           "Gen/A=%.2fx  Gen/FFTW=%.2fx\n",
           K, ns_fftw, ns_archA, ns_gen, ns_archA/ns_gen, ns_fftw/ns_gen);

    fftw_destroy_plan(plan);
    fftw_free(fftw_in);
    fftw_free(fftw_out);
    r32_aligned_free(ir);r32_aligned_free(ii);
    r32_aligned_free(or_);r32_aligned_free(oi);
}

/*==========================================================================*/

int main(void)
{
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Generated DFT-64 Kernel — Test & Benchmark               ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    int total = 0, passed = 0;

    printf("── Correctness: gen vs naive DFT ──\n");
    { const size_t Ks[] = {8, 16, 32, 64};
      for (int i = 0; i < 4; i++) { total++; passed += test_gen_vs_naive(Ks[i]); } }

    printf("\n── Correctness: gen vs Architecture A ──\n");
    { const size_t Ks[] = {8, 16, 64, 256};
      for (int i = 0; i < 4; i++) { total++; passed += test_gen_vs_archA(Ks[i]); } }

    printf("\n── Correctness: gen roundtrip ──\n");
    { const size_t Ks[] = {8, 16, 64, 256};
      for (int i = 0; i < 4; i++) { total++; passed += test_gen_roundtrip(Ks[i]); } }

    printf("\n── Correctness: gen backward vs naive IDFT ──\n");
    { const size_t Ks[] = {8, 16, 32, 64};
      for (int i = 0; i < 4; i++) { total++; passed += test_gen_bwd_vs_naive(Ks[i]); } }

    printf("\n══════════════════════════════════════════\n");
    printf("  %d/%d passed  %s\n", passed, total,
           (passed == total) ? "✓ ALL PASSED" : "✗ FAILURES");
    printf("══════════════════════════════════════════\n\n");

    if (passed != total) return 1;

    printf("── Benchmark: Gen vs Architecture A vs FFTW (ns, forward only) ──\n");
    printf("  %-8s  %-8s  %-8s  %-8s  %-9s  %-9s\n",
           "K", "FFTW", "ArchA", "Gen", "Gen/A", "Gen/FFTW");

    run_bench(8,    500, 2000);
    run_bench(16,   500, 2000);
    run_bench(32,   500, 2000);
    run_bench(64,   200, 1000);
    run_bench(128,  200, 1000);
    run_bench(256,  200, 1000);
    run_bench(512,  50,  500);
    run_bench(1024, 50,  500);
    run_bench(2048, 20,  200);
    run_bench(4096, 20,  200);

    printf("\n── Benchmark: BACKWARD — Gen native vs ArchA vs FFTW (ns) ──\n");
    printf("  %-8s  %-8s  %-8s  %-8s  %-9s  %-9s\n",
           "K", "FFTW", "ArchA", "Gen", "Gen/A", "Gen/FFTW");

    run_bench_bwd(8,    500, 2000);
    run_bench_bwd(16,   500, 2000);
    run_bench_bwd(32,   500, 2000);
    run_bench_bwd(64,   200, 1000);
    run_bench_bwd(128,  200, 1000);
    run_bench_bwd(256,  200, 1000);

    fftw_cleanup();
    return 0;
}
