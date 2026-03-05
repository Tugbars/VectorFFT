/*
 * bench_radix128_n1_packed.c — Radix-128 N1: strided vs packed vs FFTW
 */
#include "vfft_test_utils.h"
#include <fftw3.h>

#ifndef RESTRICT
#define RESTRICT __restrict__
#endif
#ifndef ALIGNAS_32
#ifdef _MSC_VER
#define ALIGNAS_32 __declspec(align(32))
#else
#define ALIGNAS_32 __attribute__((aligned(32)))
#endif
#endif

#include "fft_radix128_scalar_n1_gen.h"
#include "fft_radix128_avx2_n1_gen.h"
#include "fft_radix128_n1_packed.h"

/* Naive DFT-128 */
static void naive_dft128(size_t K, size_t k,
    const double *ir, const double *ii,
    double *or_, double *oi)
{
    for (int m = 0; m < 128; m++) {
        double sr = 0, si = 0;
        for (int n = 0; n < 128; n++) {
            double a = -2.0 * M_PI * m * n / 128.0;
            sr += ir[n*K+k]*cos(a) - ii[n*K+k]*sin(a);
            si += ir[n*K+k]*sin(a) + ii[n*K+k]*cos(a);
        }
        or_[m*K+k] = sr; oi[m*K+k] = si;
    }
}

/* Correctness: packed AVX2 vs naive */
static int test_packed_avx2(size_t K) {
    size_t T = 4, N = 128 * K;
    if (K < T || K % T != 0) return 1;

    double *ir = aa64(N), *ii_ = aa64(N), *nr = aa64(N), *ni = aa64(N);
    double *pir = aa64(N), *pii = aa64(N), *por = aa64(N), *poi = aa64(N);
    double *sor = aa64(N), *soi = aa64(N);

    fill_rand(ir, N, 1000+(unsigned)K);
    fill_rand(ii_, N, 2000+(unsigned)K);

    r128_repack_strided_to_packed(ir, ii_, pir, pii, K, T);
    r128_n1_packed_fwd_avx2(pir, pii, por, poi, K, T);
    r128_repack_packed_to_strided(por, poi, sor, soi, K, T);

    for (size_t k = 0; k < K; k++)
        naive_dft128(K, k, ir, ii_, nr, ni);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(sor[i]-nr[i]), fabs(soi[i]-ni[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(nr, N), max_abs(ni, N));
    double rel = mag > 0 ? err / mag : err;
    int pass = rel < 5e-13;
    printf("  packed T=%zu K=%-5zu rel=%.2e  %s\n", T, K, rel, pass?"PASS":"FAIL");

    r32_aligned_free(ir); r32_aligned_free(ii_); r32_aligned_free(nr); r32_aligned_free(ni);
    r32_aligned_free(pir); r32_aligned_free(pii); r32_aligned_free(por); r32_aligned_free(poi);
    r32_aligned_free(sor); r32_aligned_free(soi);
    return pass;
}

/* Benchmark */
static void run_bench(size_t K, int warm, int trials) {
    size_t N = 128 * K, T = 4;
    double *ir = aa64(N), *ii_ = aa64(N), *or_ = aa64(N), *oi = aa64(N);
    fill_rand(ir, N, 9000+(unsigned)K);
    fill_rand(ii_, N, 9500+(unsigned)K);

    /* FFTW */
    fftw_complex *fin = fftw_alloc_complex(N), *fout = fftw_alloc_complex(N);
    for (size_t k = 0; k < K; k++)
        for (int n = 0; n < 128; n++) {
            fin[k*128+n][0] = ir[n*K+k]; fin[k*128+n][1] = ii_[n*K+k];
        }
    int na[1] = {128};
    fftw_plan plan = fftw_plan_many_dft(1, na, (int)K,
        fin, NULL, 1, 128, fout, NULL, 1, 128, FFTW_FORWARD, FFTW_MEASURE);
    for (int i = 0; i < warm; i++) fftw_execute(plan);
    double bfw = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns(); fftw_execute(plan);
        double dt = get_ns() - t0; if (dt < bfw) bfw = dt;
    }

    /* Strided AVX2 */
    for (int i = 0; i < warm; i++)
        radix128_n1_dit_kernel_fwd_avx2(ir, ii_, or_, oi, K);
    double ns_str = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns();
        radix128_n1_dit_kernel_fwd_avx2(ir, ii_, or_, oi, K);
        double dt = get_ns() - t0; if (dt < ns_str) ns_str = dt;
    }

    /* Packed AVX2 */
    double ns_pk = 1e18;
    if (K >= T && K % T == 0) {
        double *pir = aa64(N), *pii = aa64(N), *por = aa64(N), *poi = aa64(N);
        r128_repack_strided_to_packed(ir, ii_, pir, pii, K, T);
        for (int i = 0; i < warm; i++)
            r128_n1_packed_fwd_avx2(pir, pii, por, poi, K, T);
        for (int t = 0; t < trials; t++) {
            double t0 = get_ns();
            r128_n1_packed_fwd_avx2(pir, pii, por, poi, K, T);
            double dt = get_ns() - t0; if (dt < ns_pk) ns_pk = dt;
        }
        r32_aligned_free(pir); r32_aligned_free(pii);
        r32_aligned_free(por); r32_aligned_free(poi);
    }

    printf("  K=%-5zu  FFTW=%7.0f  str=%7.0f(%5.2fx)  pkd=%7.0f(%5.2fx)\n",
           K, bfw, ns_str, bfw/ns_str, ns_pk, bfw/ns_pk);

    fftw_destroy_plan(plan); fftw_free(fin); fftw_free(fout);
    r32_aligned_free(ir); r32_aligned_free(ii_); r32_aligned_free(or_); r32_aligned_free(oi);
}

int main(void) {
    R32_REQUIRE_AVX2();

    printf("====================================================================\n");
    printf("  DFT-128 N1 AVX2: strided vs packed (T=4) vs FFTW\n");
    printf("====================================================================\n\n");

    int p = 0, t = 0;
    printf("-- Correctness: packed vs naive --\n");
    { size_t Ks[] = {4,8,16,32,64,128,256};
      for (int i = 0; i < 7; i++) { t++; p += test_packed_avx2(Ks[i]); } }

    printf("\n======================================\n");
    printf("  %d/%d passed  %s\n", p, t, p==t ? "ALL PASSED" : "FAILURES");
    printf("======================================\n");
    if (p != t) return 1;

    printf("\n-- BENCHMARK (ns, forward, DFT-only for packed) --\n\n");
    run_bench(4,     500, 3000);
    run_bench(8,     500, 3000);
    run_bench(16,    500, 2000);
    run_bench(32,    500, 2000);
    run_bench(64,    200, 1000);
    run_bench(128,   200, 1000);
    run_bench(256,   100, 500);
    run_bench(512,   100, 500);
    run_bench(1024,   50, 300);

    fftw_cleanup();
    return 0;
}
