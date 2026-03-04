/*
 * bench_radix32_avx2_n1.c — AVX2 N1 (twiddle-less) DFT-32 vs FFTW
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
#undef  R32AN_LD
#undef  R32AN_ST
#define R32AN_LD(p)   _mm256_load_pd(p)
#define R32AN_ST(p,v) _mm256_store_pd((p),(v))
#include "fft_radix32_avx2_n1.h"
/* ═══════════════════════════════════════════════════════════════ */

static void naive_dft32_fwd(size_t K,
    const double *ir, const double *ii,
    double *or_, double *oi) {
    for (size_t k = 0; k < K; k++) {
        double xr[32], xi[32];
        for (int n = 0; n < 32; n++) { xr[n] = ir[n*K+k]; xi[n] = ii[n*K+k]; }
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
}

static int test_fwd(size_t K) {
    size_t N = 32 * K;
    double *ir = aa(N), *ii_ = aa(N), *gr = aa(N), *gi = aa(N);
    double *nr = aa(N), *ni = aa(N);
    fill_rand(ir, N, 1000+(unsigned)K); fill_rand(ii_, N, 2000+(unsigned)K);
    radix32_n1_dit_kernel_fwd_avx2(ir, ii_, gr, gi, K);
    naive_dft32_fwd(K, ir, ii_, nr, ni);
    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(gr[i]-nr[i]), fabs(gi[i]-ni[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(nr, N), max_abs(ni, N));
    double rel = mag > 0 ? err / mag : err;
    int pass = rel < 5e-14;
    printf("  fwd K=%-5zu  rel=%.2e  %s\n", K, rel, pass?"PASS":"FAIL");
    free(ir);free(ii_);free(gr);free(gi);free(nr);free(ni);
    return pass;
}

static int test_roundtrip(size_t K) {
    size_t N = 32 * K;
    double *ir = aa(N), *ii_ = aa(N), *mr = aa(N), *mi = aa(N);
    double *rr = aa(N), *ri = aa(N);
    fill_rand(ir, N, 3000+(unsigned)K); fill_rand(ii_, N, 4000+(unsigned)K);
    radix32_n1_dit_kernel_fwd_avx2(ir, ii_, mr, mi, K);
    radix32_n1_dit_kernel_bwd_avx2(mr, mi, rr, ri, K);
    double err = 0;
    for (size_t i = 0; i < N; i++) {
        rr[i] /= 32.0; ri[i] /= 32.0;
        double e = fmax(fabs(ir[i]-rr[i]), fabs(ii_[i]-ri[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(ir, N), max_abs(ii_, N));
    double rel = mag > 0 ? err / mag : err;
    int pass = rel < 5e-15;
    printf("  rt  K=%-5zu  rel=%.2e  %s\n", K, rel, pass?"PASS":"FAIL");
    free(ir);free(ii_);free(mr);free(mi);free(rr);free(ri);
    return pass;
}

static int test_parseval(size_t K) {
    size_t N = 32 * K;
    double *ir = aa(N), *ii_ = aa(N), *or_ = aa(N), *oi = aa(N);
    fill_rand(ir, N, 7000+(unsigned)K); fill_rand(ii_, N, 8000+(unsigned)K);
    radix32_n1_dit_kernel_fwd_avx2(ir, ii_, or_, oi, K);
    double e_in = 0, e_out = 0;
    for (size_t i = 0; i < N; i++) {
        e_in += ir[i]*ir[i] + ii_[i]*ii_[i];
        e_out += or_[i]*or_[i] + oi[i]*oi[i];
    }
    double ratio = e_out / (32.0 * e_in);
    double err = fabs(ratio - 1.0);
    int pass = err < 1e-12;
    printf("  parseval K=%-5zu  ratio=%.14f  err=%.2e  %s\n", K, ratio, err, pass?"PASS":"FAIL");
    free(ir);free(ii_);free(or_);free(oi);
    return pass;
}

__attribute__((target("avx2,fma")))
static void run_bench(size_t K, int warm, int trials) {
    size_t N = 32 * K;
    double *ir = aa(N), *ii_ = aa(N), *or_ = aa(N), *oi = aa(N);
    fill_rand(ir, N, 9000+(unsigned)K); fill_rand(ii_, N, 9500+(unsigned)K);

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

    for (int i = 0; i < warm; i++)
        radix32_n1_dit_kernel_fwd_avx2(ir, ii_, or_, oi, K);
    double ns = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns();
        radix32_n1_dit_kernel_fwd_avx2(ir, ii_, or_, oi, K);
        double dt = get_ns() - t0; if (dt < ns) ns = dt;
    }

    printf("  K=%-5zu  FFTW=%7.0f  N1=%7.0f  N1/FFTW=%5.2fx\n", K, bfw, ns, bfw/ns);

    fftw_destroy_plan(plan); fftw_free(fin); fftw_free(fout);
    free(ir);free(ii_);free(or_);free(oi);
}

int main(void) {
    printf("====================================================================\n");
    printf("  DFT-32 AVX2 N1 (twiddle-less) vs FFTW\n");
    printf("  k-step=4, 16 YMM, 8×4 decomposition, zero twiddle loads\n");
    printf("====================================================================\n\n");

    int p = 0, t = 0;

    printf("-- Forward vs naive --\n");
    { size_t Ks[] = {4,8,16,32,64,128,256,512,1024};
      for (int i = 0; i < 9; i++) { t++; p += test_fwd(Ks[i]); } }

    printf("\n-- Roundtrip (fwd -> bwd -> scale) --\n");
    { size_t Ks[] = {4,8,16,32,64,128,256};
      for (int i = 0; i < 7; i++) { t++; p += test_roundtrip(Ks[i]); } }

    printf("\n-- Parseval (energy conservation) --\n");
    { size_t Ks[] = {4,8,16,32,64,128,256};
      for (int i = 0; i < 7; i++) { t++; p += test_parseval(Ks[i]); } }

    printf("\n======================================\n");
    printf("  %d/%d passed  %s\n", p, t, p == t ? "ALL PASSED" : "FAILURES");
    printf("======================================\n");
    if (p != t) return 1;

    printf("\n-- BENCHMARK (ns, forward, N1 no-twiddle vs FFTW batched DFT-32) --\n\n");
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
