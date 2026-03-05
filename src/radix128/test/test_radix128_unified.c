/*
 * test_radix128_unified.c — Test for the unified DFT-128 N1 header
 * Exercises scalar, AVX2, AVX-512, cross-ISA, and the auto-dispatch API.
 */
#include "vfft_test_utils.h"
#include <assert.h>
#include <stdint.h>
#include <fftw3.h>

#ifndef RESTRICT
#define RESTRICT __restrict__
#endif
#ifndef TARGET_AVX512
#define TARGET_AVX512 __attribute__((target("avx512f,avx512dq,fma")))
#endif
#ifndef ALIGNAS_64
#define ALIGNAS_64 __attribute__((aligned(64)))
#endif
#ifndef ALIGNAS_32
#define ALIGNAS_32 __attribute__((aligned(32)))
#endif

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

static const char *isa_for_K_128(size_t K) {
#ifdef __AVX512F__
    if (K >= 8 && (K & 7) == 0) return "avx512";
#endif
    if (K >= 4 && (K & 3) == 0) return "avx2";
    return "scalar";
}

/* ── Correctness: unified API vs naive ── */

static int test_unified_fwd(size_t K) {
    const size_t N = 128 * K;
    double *ir=aa64(N), *ii=aa64(N), *gr=aa64(N), *gi=aa64(N), *nr=aa64(N), *ni=aa64(N);
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
    printf("  unified fwd  K=%-4zu  isa=%-7s rel=%.2e  %s\n",
           K, isa_for_K_128(K), rel, pass?"PASS":"FAIL");
    r32_aligned_free(ir);r32_aligned_free(ii);r32_aligned_free(gr);
    r32_aligned_free(gi);r32_aligned_free(nr);r32_aligned_free(ni);
    return pass;
}

static int test_unified_bwd(size_t K) {
    const size_t N = 128 * K;
    double *ir=aa64(N), *ii=aa64(N), *gr=aa64(N), *gi=aa64(N), *nr=aa64(N), *ni=aa64(N);
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
    printf("  unified bwd  K=%-4zu  isa=%-7s rel=%.2e  %s\n",
           K, isa_for_K_128(K), rel, pass?"PASS":"FAIL");
    r32_aligned_free(ir);r32_aligned_free(ii);r32_aligned_free(gr);
    r32_aligned_free(gi);r32_aligned_free(nr);r32_aligned_free(ni);
    return pass;
}

static int test_unified_roundtrip(size_t K) {
    const size_t N = 128 * K;
    double *ir=aa64(N), *ii=aa64(N), *fr=aa64(N), *fi=aa64(N), *br=aa64(N), *bi=aa64(N);
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
    r32_aligned_free(ir);r32_aligned_free(ii);r32_aligned_free(fr);
    r32_aligned_free(fi);r32_aligned_free(br);r32_aligned_free(bi);
    return pass;
}

/* ── Cross-ISA consistency ── */

static int test_cross_isa(size_t K) {
    const size_t N = 128 * K;
    double *ir=aa64(N), *ii=aa64(N);
    double *sr=aa64(N), *si=aa64(N);
    double *ar=aa64(N), *ai=aa64(N);
    double *zr=aa64(N), *zi=aa64(N);
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

    int pass_sa = (err_sa < 1e-13);
    int pass_sz = (err_sz < 1e-13);

#ifdef __AVX512F__
    printf("  cross K=%-4zu  S<->A=%.2e  S<->Z=%.2e  %s\n",
           K, err_sa, err_sz,
           (pass_sa && pass_sz) ? "PASS" : "FAIL");
    int pass = pass_sa && pass_sz;
#else
    printf("  cross K=%-4zu  S<->A=%.2e  %s\n",
           K, err_sa, pass_sa ? "PASS" : "FAIL");
    int pass = pass_sa;
#endif

    r32_aligned_free(ir);r32_aligned_free(ii);r32_aligned_free(sr);r32_aligned_free(si);
    r32_aligned_free(ar);r32_aligned_free(ai);r32_aligned_free(zr);r32_aligned_free(zi);
    return pass;
}

/* ── Benchmarks ── */

static void run_bench(size_t K, int warmup, int trials) {
    const size_t N = 128 * K;
    double *ir=aa64(N), *ii=aa64(N), *or_=aa64(N), *oi=aa64(N);
    fill_rand(ir, N, 80000+(unsigned)K);
    fill_rand(ii, N, 90000+(unsigned)K);

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

    for (int i = 0; i < warmup; i++)
        fft_radix128_n1_forward(K, ir, ii, or_, oi);
    double best_gen = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns();
        fft_radix128_n1_forward(K, ir, ii, or_, oi);
        double dt = get_ns() - t0;
        if (dt < best_gen) best_gen = dt;
    }

    printf("  K=%-5zu [%s]  FFTW=%8.0f  Gen=%8.0f  Gen/FFTW=%.2fx\n",
           K, isa_for_K_128(K), best_fftw, best_gen, best_fftw/best_gen);

    fftw_destroy_plan(plan);
    fftw_free(fftw_in);
    fftw_free(fftw_out);
    r32_aligned_free(ir);r32_aligned_free(ii);r32_aligned_free(or_);r32_aligned_free(oi);
}

int main(void) {
    R32_REQUIRE_AVX2();

    printf("====================================================================\n");
    printf("  DFT-128 N1 — Unified Header Test\n");
    printf("====================================================================\n\n");

    int passed = 0, total = 0;

    printf("-- Unified forward (hits all ISA paths) --\n");
    { size_t Ks[] = {1, 2, 3, 4, 5, 7, 8, 12, 16, 24, 32, 64};
      for (int i = 0; i < 12; i++) { total++; passed += test_unified_fwd(Ks[i]); } }

    printf("\n-- Unified backward --\n");
    { size_t Ks[] = {1, 3, 4, 8, 16, 32};
      for (int i = 0; i < 6; i++) { total++; passed += test_unified_bwd(Ks[i]); } }

    printf("\n-- Unified roundtrip --\n");
    { size_t Ks[] = {1, 2, 4, 8, 16, 32, 64};
      for (int i = 0; i < 7; i++) { total++; passed += test_unified_roundtrip(Ks[i]); } }

    printf("\n-- Cross-ISA consistency --\n");
    { size_t Ks[] = {8, 16, 32, 64};
      for (int i = 0; i < 4; i++) { total++; passed += test_cross_isa(Ks[i]); } }

    printf("\n======================================\n");
    printf("  %d/%d passed  %s\n", passed, total,
           passed==total ? "ALL PASSED" : "FAILURES");
    printf("======================================\n");

    if (passed != total) return 1;

    printf("\n-- Benchmark: unified API vs FFTW (ns, fwd) --\n");
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