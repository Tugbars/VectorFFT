/*
 * bench_radix32_u2.c — Test + benchmark: fft_radix32_avx512_n1_u2.h
 *
 * Tests U=1 and U=2 pipelined N1 (twiddle-less) DFT-32 kernels.
 * U=1: k-step=8, single pipeline.
 * U=2: k-step=16, dual interleaved pipelines, prefetch.
 *
 * Production headers tested:
 *   fft_radix32_avx512_n1_u2.h  (U1 + U2 × fwd + bwd = 4 kernels)
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
    if (posix_memalign(&p, 64, n * sizeof(double)) != 0) abort();
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
#include "fft_radix32_avx512_n1_u2.h"
/* ═══════════════════════════════════════════════════════════════ */

/* Naive DFT-32 reference (no twiddles) */
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
static void naive_dft32_bwd(size_t K,
    const double *ir, const double *ii,
    double *or_, double *oi) {
    for (size_t k = 0; k < K; k++) {
        double xr[32], xi[32];
        for (int n = 0; n < 32; n++) { xr[n] = ir[n*K+k]; xi[n] = ii[n*K+k]; }
        for (int m = 0; m < 32; m++) {
            double sr = 0, si = 0;
            for (int n = 0; n < 32; n++) {
                double a = +2.0 * M_PI * m * n / 32.0;
                sr += xr[n]*cos(a) - xi[n]*sin(a);
                si += xr[n]*sin(a) + xi[n]*cos(a);
            }
            or_[m*K + k] = sr; oi[m*K + k] = si;
        }
    }
}

typedef void (*kern_fn)(const double*, const double*, double*, double*, size_t);

/* ── Forward vs naive ── */
static int test_fwd(const char *lbl, kern_fn fn, size_t K) {
    size_t N = 32 * K;
    double *ir = aa(N), *ii_ = aa(N), *gr = aa(N), *gi = aa(N);
    double *nr = aa(N), *ni = aa(N);
    fill_rand(ir, N, 1000+(unsigned)K); fill_rand(ii_, N, 2000+(unsigned)K);
    fn(ir, ii_, gr, gi, K);
    naive_dft32_fwd(K, ir, ii_, nr, ni);
    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(gr[i]-nr[i]), fabs(gi[i]-ni[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(nr, N), max_abs(ni, N));
    double rel = mag > 0 ? err / mag : err;
    int pass = rel < 5e-14;
    printf("  %-5s fwd K=%-5zu  rel=%.2e  %s\n", lbl, K, rel, pass?"PASS":"FAIL");
    free(ir);free(ii_);free(gr);free(gi);free(nr);free(ni);
    return pass;
}

/* ── Roundtrip: fwd → bwd → scale ── */
static int test_rt(const char *lbl, kern_fn fwd, kern_fn bwd, size_t K) {
    size_t N = 32 * K;
    double *ir = aa(N), *ii_ = aa(N), *mr = aa(N), *mi = aa(N), *rr = aa(N), *ri = aa(N);
    fill_rand(ir, N, 3000+(unsigned)K); fill_rand(ii_, N, 4000+(unsigned)K);
    fwd(ir, ii_, mr, mi, K);
    bwd(mr, mi, rr, ri, K);
    double err = 0;
    for (size_t i = 0; i < N; i++) {
        rr[i] /= 32.0; ri[i] /= 32.0;
        double e = fmax(fabs(ir[i]-rr[i]), fabs(ii_[i]-ri[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(ir, N), max_abs(ii_, N));
    double rel = mag > 0 ? err / mag : err;
    int pass = rel < 5e-15;
    printf("  %-5s rt  K=%-5zu  rel=%.2e  %s\n", lbl, K, rel, pass?"PASS":"FAIL");
    free(ir);free(ii_);free(mr);free(mi);free(rr);free(ri);
    return pass;
}

/* ── Cross: U1 ↔ U2 (should be bit-exact: same algorithm, different pipelining) ── */
static int test_cross(size_t K) {
    size_t N = 32 * K;
    double *ir = aa(N), *ii_ = aa(N);
    double *ar = aa(N), *ai = aa(N), *br = aa(N), *bi = aa(N);
    fill_rand(ir, N, 5000+(unsigned)K); fill_rand(ii_, N, 6000+(unsigned)K);
    radix32_n1_dit_kernel_fwd_avx512_u1(ir, ii_, ar, ai, K);
    radix32_n1_dit_kernel_fwd_avx512_u2(ir, ii_, br, bi, K);
    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(ar[i]-br[i]), fabs(ai[i]-bi[i]));
        if (e > err) err = e;
    }
    int pass = (err == 0.0);
    printf("  U1<->U2  K=%-5zu  maxdiff=%.2e  %s%s\n",
           K, err, pass?"PASS":"FAIL", pass?" (bit-exact)":"");
    free(ir);free(ii_);free(ar);free(ai);free(br);free(bi);
    return pass;
}

/* ── Benchmark ── */
__attribute__((target("avx512f,avx512dq,fma")))
static void run_bench(size_t K, int warm, int trials) {
    size_t N = 32 * K;
    double *ir = aa(N), *ii_ = aa(N), *or_ = aa(N), *oi = aa(N);
    fill_rand(ir, N, 9000+(unsigned)K); fill_rand(ii_, N, 9500+(unsigned)K);

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
        double t0 = get_ns(); fftw_execute(plan); double dt = get_ns()-t0;
        if (dt < bfw) bfw = dt;
    }

    /* U1 */
    for (int i = 0; i < warm; i++)
        radix32_n1_dit_kernel_fwd_avx512_u1(ir, ii_, or_, oi, K);
    double ns_u1 = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns();
        radix32_n1_dit_kernel_fwd_avx512_u1(ir, ii_, or_, oi, K);
        double dt = get_ns()-t0; if (dt < ns_u1) ns_u1 = dt;
    }

    /* U2 (K >= 16 required) */
    double ns_u2 = 1e18;
    if (K >= 16) {
        for (int i = 0; i < warm; i++)
            radix32_n1_dit_kernel_fwd_avx512_u2(ir, ii_, or_, oi, K);
        for (int t = 0; t < trials; t++) {
            double t0 = get_ns();
            radix32_n1_dit_kernel_fwd_avx512_u2(ir, ii_, or_, oi, K);
            double dt = get_ns()-t0; if (dt < ns_u2) ns_u2 = dt;
        }
    }

    printf("  K=%-5zu  FFTW=%7.0f  U1=%7.0f(%5.2fx)", K, bfw, ns_u1, bfw/ns_u1);
    if (K >= 16)
        printf("  U2=%7.0f(%5.2fx)", ns_u2, bfw/ns_u2);
    printf("\n");

    fftw_destroy_plan(plan); fftw_free(fin); fftw_free(fout);
    free(ir);free(ii_);free(or_);free(oi);
}

int main(void) {
    printf("====================================================================\n");
    printf("  DFT-32 AVX-512 N1: U=1 + U=2 pipelining (n1_u2.h)\n");
    printf("====================================================================\n\n");

    int p = 0, t = 0;

    printf("-- U1 forward vs naive --\n");
    { size_t Ks[] = {8,16,32,64,128,256,512,1024};
      for (int i = 0; i < 8; i++) { t++; p += test_fwd("U1", radix32_n1_dit_kernel_fwd_avx512_u1, Ks[i]); } }

    printf("\n-- U2 forward vs naive --\n");
    { size_t Ks[] = {16,32,64,128,256,512,1024};
      for (int i = 0; i < 7; i++) { t++; p += test_fwd("U2", radix32_n1_dit_kernel_fwd_avx512_u2, Ks[i]); } }

    printf("\n-- Roundtrip (fwd -> bwd -> scale) --\n");
    { size_t Ks[] = {8,16,32,64,128,256};
      for (int i = 0; i < 6; i++) {
          t++; p += test_rt("U1", radix32_n1_dit_kernel_fwd_avx512_u1, radix32_n1_dit_kernel_bwd_avx512_u1, Ks[i]);
      }
      for (int i = 1; i < 6; i++) {  /* U2: K >= 16 */
          t++; p += test_rt("U2", radix32_n1_dit_kernel_fwd_avx512_u2, radix32_n1_dit_kernel_bwd_avx512_u2, Ks[i]);
      }
    }

    printf("\n-- Cross: U1 <-> U2 (bit-exact?) --\n");
    { size_t Ks[] = {16,32,64,128,256,512};
      for (int i = 0; i < 6; i++) { t++; p += test_cross(Ks[i]); } }

    printf("\n======================================\n");
    printf("  %d/%d passed  %s\n", p, t, p == t ? "ALL PASSED" : "FAILURES");
    printf("======================================\n");
    if (p != t) return 1;

    printf("\n-- BENCHMARK (ns, forward, N1 no-twiddle) --\n\n");
    run_bench(8,    500, 5000);
    run_bench(16,   500, 5000);
    run_bench(32,   500, 3000);
    run_bench(64,   500, 3000);
    run_bench(128,  200, 2000);
    run_bench(256,  200, 2000);
    run_bench(512,  100, 1000);
    run_bench(1024, 100, 1000);

    fftw_cleanup();
    return 0;
}
