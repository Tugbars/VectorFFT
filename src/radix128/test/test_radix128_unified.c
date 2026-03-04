/*
 * test_radix128_unified.c — Test for the unified DFT-128 N1 header
 * Exercises scalar, AVX2, AVX-512, cross-ISA, and the auto-dispatch API.
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

#define TARGET_AVX512 __attribute__((target("avx512f,avx512dq,fma")))
#define RESTRICT __restrict__
#define ALIGNAS_64 __attribute__((aligned(64)))
#define ALIGNAS_32 __attribute__((aligned(32)))

static void *aligned_alloc_64(size_t size) {
    void *p = NULL;
    posix_memalign(&p, 64, size);
    memset(p, 0, size);
    return p;
}
#define aa(n) (double*)aligned_alloc_64((n)*sizeof(double))

/* The unified header — pulls everything in */
#include "fft_radix128_n1.h"

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

/* ── Correctness: unified API vs naive ── */

static int test_unified_fwd(size_t K) {
    const size_t N = 128 * K;
    double *ir=aa(N), *ii=aa(N), *gr=aa(N), *gi=aa(N), *nr=aa(N), *ni=aa(N);
    fill_rand(ir, N, 10000+(unsigned)K);
    fill_rand(ii, N, 20000+(unsigned)K);

    fft_radix128_n1_forward(K, ir, ii, gr, gi);
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

    const char *isa = "?";
#ifdef __AVX512F__
    if (K >= 8 && (K & 7) == 0) isa = "avx512";
    else
#endif
    if (K >= 4 && (K & 3) == 0) isa = "avx2";
    else isa = "scalar";

    printf("  unified fwd  K=%-4zu  isa=%-7s rel=%.2e  %s\n",
           K, isa, rel, pass?"PASS":"FAIL");
    free(ir);free(ii);free(gr);free(gi);free(nr);free(ni);
    return pass;
}

static int test_unified_bwd(size_t K) {
    const size_t N = 128 * K;
    double *ir=aa(N), *ii=aa(N), *gr=aa(N), *gi=aa(N), *nr=aa(N), *ni=aa(N);
    fill_rand(ir, N, 30000+(unsigned)K);
    fill_rand(ii, N, 40000+(unsigned)K);

    fft_radix128_n1_backward(K, ir, ii, gr, gi);
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

    const char *isa = "?";
#ifdef __AVX512F__
    if (K >= 8 && (K & 7) == 0) isa = "avx512";
    else
#endif
    if (K >= 4 && (K & 3) == 0) isa = "avx2";
    else isa = "scalar";

    printf("  unified bwd  K=%-4zu  isa=%-7s rel=%.2e  %s\n",
           K, isa, rel, pass?"PASS":"FAIL");
    free(ir);free(ii);free(gr);free(gi);free(nr);free(ni);
    return pass;
}

static int test_unified_roundtrip(size_t K) {
    const size_t N = 128 * K;
    double *ir=aa(N), *ii=aa(N), *fr=aa(N), *fi=aa(N), *br=aa(N), *bi=aa(N);
    fill_rand(ir, N, 50000+(unsigned)K);
    fill_rand(ii, N, 60000+(unsigned)K);

    fft_radix128_n1_forward(K, ir, ii, fr, fi);
    fft_radix128_n1_backward(K, fr, fi, br, bi);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(br[i]-128.0*ir[i]), fabs(bi[i]-128.0*ii[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(ir,N), max_abs(ii,N));
    double rel = (mag > 0) ? err / (128.0*mag) : err;
    int pass = (rel < 1e-13);

    printf("  unified rt   K=%-4zu  rel=%.2e  %s\n", K, rel, pass?"PASS":"FAIL");
    free(ir);free(ii);free(fr);free(fi);free(br);free(bi);
    return pass;
}

/* ── Cross-ISA consistency: per-ISA API returns same results ── */

static int test_cross_isa(size_t K) {
    /* K must be >= 8 and multiple of 8 so all three ISAs can run */
    const size_t N = 128 * K;
    double *ir=aa(N), *ii=aa(N);
    double *sr=aa(N), *si=aa(N);  /* scalar */
    double *ar=aa(N), *ai=aa(N);  /* avx2 */
    double *zr=aa(N), *zi=aa(N);  /* avx512 */
    fill_rand(ir, N, 70000+(unsigned)K);
    fill_rand(ii, N, 71000+(unsigned)K);

    radix128_n1_forward_scalar(K, ir, ii, sr, si);
    radix128_n1_forward_avx2(K, ir, ii, ar, ai);
#ifdef __AVX512F__
    radix128_n1_forward_avx512(K, ir, ii, zr, zi);
#endif

    double err_sa = 0, err_sz = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(sr[i]-ar[i]), fabs(si[i]-ai[i]));
        if (e > err_sa) err_sa = e;
#ifdef __AVX512F__
        e = fmax(fabs(sr[i]-zr[i]), fabs(si[i]-zi[i]));
        if (e > err_sz) err_sz = e;
#endif
    }

    /* FMA rounding gives ~2-4 ULP diff between ISAs */
    int pass_sa = (err_sa < 1e-13);
    int pass_sz = (err_sz < 1e-13);

#ifdef __AVX512F__
    printf("  cross K=%-4zu  S↔A=%.2e  S↔Z=%.2e  %s\n",
           K, err_sa, err_sz,
           (pass_sa && pass_sz) ? "PASS" : "FAIL");
    int pass = pass_sa && pass_sz;
#else
    printf("  cross K=%-4zu  S↔A=%.2e  %s\n",
           K, err_sa, pass_sa ? "PASS" : "FAIL");
    int pass = pass_sa;
#endif

    free(ir);free(ii);free(sr);free(si);free(ar);free(ai);
    free(zr);free(zi);
    return pass;
}

/* ── Benchmarks: unified API vs FFTW ── */

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

    /* Unified API */
    for (int i = 0; i < warmup; i++)
        fft_radix128_n1_forward(K, ir, ii, or_, oi);
    double best_gen = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns();
        fft_radix128_n1_forward(K, ir, ii, or_, oi);
        double dt = get_ns() - t0;
        if (dt < best_gen) best_gen = dt;
    }

    const char *isa = "?";
#ifdef __AVX512F__
    if (K >= 8 && (K & 7) == 0) isa = "z512";
    else
#endif
    if (K >= 4 && (K & 3) == 0) isa = "avx2";
    else isa = "sclr";

    printf("  K=%-5zu [%s]  FFTW=%8.0f  Gen=%8.0f  Gen/FFTW=%.2fx\n",
           K, isa, best_fftw, best_gen, best_fftw/best_gen);

    fftw_destroy_plan(plan);
    fftw_free(fftw_in);
    fftw_free(fftw_out);
    free(ir);free(ii);free(or_);free(oi);
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  DFT-128 N1 — Unified Header Test                         ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    int passed = 0, total = 0;

    /* Forward: test every dispatch path */
    printf("── Unified forward (hits all ISA paths) ──\n");
    { size_t Ks[] = {1, 2, 3, 4, 5, 7, 8, 12, 16, 24, 32, 64};
      for (int i = 0; i < 12; i++) { total++; passed += test_unified_fwd(Ks[i]); } }

    /* Backward */
    printf("\n── Unified backward ──\n");
    { size_t Ks[] = {1, 3, 4, 8, 16, 32};
      for (int i = 0; i < 6; i++) { total++; passed += test_unified_bwd(Ks[i]); } }

    /* Roundtrip */
    printf("\n── Unified roundtrip ──\n");
    { size_t Ks[] = {1, 2, 4, 8, 16, 32, 64};
      for (int i = 0; i < 7; i++) { total++; passed += test_unified_roundtrip(Ks[i]); } }

    /* Cross-ISA (K divisible by 8 so all ISAs run) */
    printf("\n── Cross-ISA consistency ──\n");
    { size_t Ks[] = {8, 16, 32, 64};
      for (int i = 0; i < 4; i++) { total++; passed += test_cross_isa(Ks[i]); } }

    printf("\n══════════════════════════════════════════\n");
    printf("  %d/%d passed  %s\n", passed, total,
           passed==total ? "✓ ALL PASSED" : "✗ FAILURES");
    printf("══════════════════════════════════════════\n");

    if (passed != total) return 1;

    /* Benchmarks */
    printf("\n── Benchmark: unified API vs FFTW (ns, fwd) ──\n");

    run_bench(1,    500, 3000);
    run_bench(3,    500, 3000);
    run_bench(4,    500, 3000);
    run_bench(8,    500, 2000);
    run_bench(16,   500, 2000);
    run_bench(32,   200, 1000);
    run_bench(64,   200, 1000);
    run_bench(128,  100, 500);
    run_bench(256,  100, 500);

    fftw_cleanup();
    return 0;
}