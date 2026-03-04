/*
 * bench_radix32_avx2_tw_unified.c — AVX2 unified twiddled DFT-32
 *
 * Tests both packed (production) and strided (fallback) paths.
 * Production: fft_radix32_avx2_tw_unified.h
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void *aa(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 32, n * sizeof(double)) != 0) abort();
    memset(p, 0, n * sizeof(double));
    return p;
}
static void fill_rand(double *p, size_t n, unsigned s) {
    srand(s);
    for (size_t i = 0; i < n; i++) p[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
}
static double max_abs(const double *p, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; i++) { double a = fabs(p[i]); if (a > m) m = a; }
    return m;
}
static double get_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

/* ═══════════════════════════════════════════════════════════════ */
#include "fft_radix32_avx2_tw_unified.h"
/* ═══════════════════════════════════════════════════════════════ */

static void naive_tw_dft32_fwd(size_t K, size_t k,
    const double *ir, const double *ii,
    const double *twr, const double *twi,
    double *or_, double *oi) {
    double xr[32], xi[32];
    xr[0] = ir[k]; xi[0] = ii[k];
    for (int n = 1; n < 32; n++) {
        double wr = twr[(n-1)*K + k], wi = twi[(n-1)*K + k];
        xr[n] = ir[n*K + k]*wr - ii[n*K + k]*wi;
        xi[n] = ir[n*K + k]*wi + ii[n*K + k]*wr;
    }
    for (int m = 0; m < 32; m++) {
        double sr = 0, si = 0;
        for (int n = 0; n < 32; n++) {
            double a = -2.0 * M_PI * m * n / 32.0;
            sr += xr[n]*cos(a) - xi[n]*sin(a);
            si += xr[n]*sin(a) + xi[n]*cos(a);
        }
        or_[m*K + k] = sr; oi[m*K + k] = si;
    }
}

/* ── Packed path vs naive ── */
static int test_packed(size_t K) {
    size_t N = 32 * K, T = r32a_packed_optimal_T(K);
    double *ir = aa(N), *ii_ = aa(N), *nr = aa(N), *ni = aa(N);
    double *ftwr = aa(31*K), *ftwi = aa(31*K);
    double *ptwr = aa(31*K), *ptwi = aa(31*K);
    double *pir = aa(N), *pii = aa(N), *por = aa(N), *poi = aa(N);
    double *sor = aa(N), *soi = aa(N);

    fill_rand(ir, N, 1000+(unsigned)K);
    fill_rand(ii_, N, 2000+(unsigned)K);
    r32a_build_flat_twiddles(K, -1, ftwr, ftwi);
    r32a_build_packed_twiddles(K, T, ftwr, ftwi, ptwr, ptwi);
    r32a_pack_input(ir, ii_, pir, pii, K, T);

    radix32_tw_packed_dispatch_fwd_avx2(K, pir, pii, por, poi, ptwr, ptwi);
    r32a_unpack_output(por, poi, sor, soi, K, T);

    for (size_t k = 0; k < K; k++)
        naive_tw_dft32_fwd(K, k, ir, ii_, ftwr, ftwi, nr, ni);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(sor[i]-nr[i]), fabs(soi[i]-ni[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(nr, N), max_abs(ni, N));
    double rel = mag > 0 ? err / mag : err;
    int pass = rel < 5e-13;
    printf("  packed fwd K=%-5zu T=%-3zu rel=%.2e  %s\n", K, T, rel, pass?"PASS":"FAIL");

    free(ir);free(ii_);free(nr);free(ni);free(ftwr);free(ftwi);
    free(ptwr);free(ptwi);free(pir);free(pii);free(por);free(poi);
    free(sor);free(soi);
    return pass;
}

/* ── Strided path vs naive ── */
static int test_strided(size_t K) {
    size_t N = 32 * K;
    double *ir = aa(N), *ii_ = aa(N), *or_ = aa(N), *oi = aa(N);
    double *nr = aa(N), *ni = aa(N);
    double *ftwr = aa(31*K), *ftwi = aa(31*K);

    fill_rand(ir, N, 3000+(unsigned)K);
    fill_rand(ii_, N, 4000+(unsigned)K);
    r32a_build_flat_twiddles(K, -1, ftwr, ftwi);

    radix32_tw_strided_dispatch_fwd_avx2(K, ir, ii_, or_, oi, ftwr, ftwi);

    for (size_t k = 0; k < K; k++)
        naive_tw_dft32_fwd(K, k, ir, ii_, ftwr, ftwi, nr, ni);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(or_[i]-nr[i]), fabs(oi[i]-ni[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(nr, N), max_abs(ni, N));
    double rel = mag > 0 ? err / mag : err;
    int pass = rel < 5e-13;
    printf("  strided fwd K=%-5zu rel=%.2e  %s\n", K, rel, pass?"PASS":"FAIL");

    free(ir);free(ii_);free(or_);free(oi);free(nr);free(ni);
    free(ftwr);free(ftwi);
    return pass;
}

/* ── Cross: packed ↔ strided ── */
static int test_cross(size_t K) {
    size_t N = 32 * K, T = r32a_packed_optimal_T(K);
    double *ir = aa(N), *ii_ = aa(N);
    double *ftwr = aa(31*K), *ftwi = aa(31*K);
    double *ptwr = aa(31*K), *ptwi = aa(31*K);

    fill_rand(ir, N, 5000+(unsigned)K);
    fill_rand(ii_, N, 6000+(unsigned)K);
    r32a_build_flat_twiddles(K, -1, ftwr, ftwi);
    r32a_build_packed_twiddles(K, T, ftwr, ftwi, ptwr, ptwi);

    /* Strided */
    double *sr = aa(N), *si = aa(N);
    radix32_tw_strided_dispatch_fwd_avx2(K, ir, ii_, sr, si, ftwr, ftwi);

    /* Packed */
    double *pir = aa(N), *pii = aa(N), *por = aa(N), *poi = aa(N);
    r32a_pack_input(ir, ii_, pir, pii, K, T);
    radix32_tw_packed_dispatch_fwd_avx2(K, pir, pii, por, poi, ptwr, ptwi);
    double *pr = aa(N), *pi_ = aa(N);
    r32a_unpack_output(por, poi, pr, pi_, K, T);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(sr[i]-pr[i]), fabs(si[i]-pi_[i]));
        if (e > err) err = e;
    }
    int pass = (err == 0.0);
    printf("  packed<->strided K=%-5zu T=%-3zu maxdiff=%.2e  %s%s\n",
           K, T, err, pass?"PASS":"FAIL", pass?" (bit-exact)":"");

    free(ir);free(ii_);free(sr);free(si);
    free(ftwr);free(ftwi);free(ptwr);free(ptwi);
    free(pir);free(pii);free(por);free(poi);free(pr);free(pi_);
    return pass;
}

/* ── Benchmark ── */
__attribute__((target("avx2,fma")))
static void run_bench(size_t K, int warm, int trials) {
    size_t N = 32 * K, T = r32a_packed_optimal_T(K);
    double *ir = aa(N), *ii_ = aa(N), *or_ = aa(N), *oi = aa(N);
    double *ftwr = aa(31*K), *ftwi = aa(31*K);
    double *ptwr = aa(31*K), *ptwi = aa(31*K);
    double *pir = aa(N), *pii = aa(N), *por = aa(N), *poi = aa(N);

    fill_rand(ir, N, 9000+(unsigned)K);
    fill_rand(ii_, N, 9500+(unsigned)K);
    r32a_build_flat_twiddles(K, -1, ftwr, ftwi);
    r32a_build_packed_twiddles(K, T, ftwr, ftwi, ptwr, ptwi);
    r32a_pack_input(ir, ii_, pir, pii, K, T);

    /* FFTW */
    fftw_complex *fin = fftw_alloc_complex(N), *fout = fftw_alloc_complex(N);
    for (size_t k = 0; k < K; k++)
        for (int n = 0; n < 32; n++) {
            fin[k*32+n][0] = ir[n*K+k]; fin[k*32+n][1] = ii_[n*K+k];
        }
    int na[1] = {32};
    fftw_plan plan = fftw_plan_many_dft(1, na, (int)K,
        fin, NULL, 1, 32, fout, NULL, 1, 32, FFTW_FORWARD, FFTW_MEASURE);
    for (int i = 0; i < warm; i++) fftw_execute(plan);
    double bfw = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns(); fftw_execute(plan);
        double dt = get_ns() - t0; if (dt < bfw) bfw = dt;
    }

    /* Packed (production) */
    for (int i = 0; i < warm; i++)
        radix32_tw_packed_dispatch_fwd_avx2(K, pir, pii, por, poi, ptwr, ptwi);
    double ns_pk = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns();
        radix32_tw_packed_dispatch_fwd_avx2(K, pir, pii, por, poi, ptwr, ptwi);
        double dt = get_ns() - t0; if (dt < ns_pk) ns_pk = dt;
    }

    /* Strided (fallback) */
    for (int i = 0; i < warm; i++)
        radix32_tw_strided_dispatch_fwd_avx2(K, ir, ii_, or_, oi, ftwr, ftwi);
    double ns_st = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns();
        radix32_tw_strided_dispatch_fwd_avx2(K, ir, ii_, or_, oi, ftwr, ftwi);
        double dt = get_ns() - t0; if (dt < ns_st) ns_st = dt;
    }

    printf("  K=%-5zu T=%-3zu  FFTW=%7.0f  packed=%7.0f(%5.2fx)  strided=%7.0f(%5.2fx)\n",
           K, T, bfw, ns_pk, bfw/ns_pk, ns_st, bfw/ns_st);

    fftw_destroy_plan(plan); fftw_free(fin); fftw_free(fout);
    free(ir);free(ii_);free(or_);free(oi);
    free(ftwr);free(ftwi);free(ptwr);free(ptwi);
    free(pir);free(pii);free(por);free(poi);
}

int main(void) {
    printf("====================================================================\n");
    printf("  DFT-32 AVX2 TWIDDLED: UNIFIED DISPATCH TEST + BENCHMARK\n");
    printf("====================================================================\n");
    printf("  Packed T default: %d\n", R32A_PACKED_BLOCK_T);
    printf("  Fused twiddle + DFT-32 (35%% more work than FFTW baseline)\n\n");

    int p = 0, t = 0;

    printf("-- Packed path vs naive --\n");
    { size_t Ks[] = {4,8,16,32,64,128,256,512,1024};
      for (int i = 0; i < 9; i++) { t++; p += test_packed(Ks[i]); } }

    printf("\n-- Strided path vs naive --\n");
    { size_t Ks[] = {4,8,16,32,64,128,256,512};
      for (int i = 0; i < 8; i++) { t++; p += test_strided(Ks[i]); } }

    printf("\n-- Cross: packed <-> strided --\n");
    { size_t Ks[] = {4,8,16,32,64,128,256};
      for (int i = 0; i < 7; i++) { t++; p += test_cross(Ks[i]); } }

    printf("\n======================================\n");
    printf("  %d/%d passed  %s\n", p, t, p == t ? "ALL PASSED" : "FAILURES");
    printf("======================================\n");
    if (p != t) return 1;

    printf("\n-- BENCHMARK (ns, forward) --\n");
    printf("  packed = DFT-only, data pre-packed (production)\n");
    printf("  strided = data in stride-K layout (fallback)\n\n");

    run_bench(4,     500, 5000);
    run_bench(8,     500, 5000);
    run_bench(16,    500, 5000);
    run_bench(32,    500, 3000);
    run_bench(64,    500, 3000);
    run_bench(128,   200, 2000);
    run_bench(256,   200, 2000);
    run_bench(512,   100, 1000);
    run_bench(1024,  100, 1000);
    run_bench(2048,   50, 500);
    run_bench(4096,   50, 500);

    fftw_cleanup();
    return 0;
}
