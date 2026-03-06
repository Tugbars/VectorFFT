/*
 * test_planner.c — Verify vfft_planner for a wide range of N values
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "vfft_planner.h"

static double *aa64(size_t n) {
    double *p = (double *)vfft_aligned_alloc(64, n * sizeof(double));
    memset(p, 0, n * sizeof(double));
    return p;
}

static void fill_rand(double *p, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++)
        p[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
}

static void naive_dft(const double *ir, const double *ii,
                       double *nr, double *ni, size_t N) {
    for (size_t m = 0; m < N; m++) {
        double sr = 0, si = 0;
        for (size_t n = 0; n < N; n++) {
            double a = -2.0 * M_PI * m * n / (double)N;
            sr += ir[n]*cos(a) - ii[n]*sin(a);
            si += ir[n]*sin(a) + ii[n]*cos(a);
        }
        nr[m] = sr;
        ni[m] = si;
    }
}

static int test_fwd(size_t N, const vfft_codelet_registry *reg, int verbose) {
    double *ir = aa64(N), *ii_ = aa64(N);
    double *gr = aa64(N), *gi = aa64(N);
    double *nr = aa64(N), *ni = aa64(N);
    fill_rand(ir, N, 1000+(unsigned)N);
    fill_rand(ii_, N, 2000+(unsigned)N);

    vfft_plan *plan = vfft_plan_create(N, reg);
    if (!plan) {
        printf("  N=%-6zu PLAN CREATION FAILED\n", N);
        vfft_aligned_free(ir);vfft_aligned_free(ii_);vfft_aligned_free(gr);vfft_aligned_free(gi);vfft_aligned_free(nr);vfft_aligned_free(ni);
        return 0;
    }

    if (verbose) vfft_plan_print(plan);

    vfft_execute_fwd(plan, ir, ii_, gr, gi);
    naive_dft(ir, ii_, nr, ni, N);

    double err = 0, mag = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(gr[i]-nr[i]), fabs(gi[i]-ni[i]));
        if (e > err) err = e;
        double m = fmax(fabs(nr[i]), fabs(ni[i]));
        if (m > mag) mag = m;
    }
    double rel = mag > 0 ? err/mag : err;

    /* Tolerance scales with log2(N) for multi-stage */
    double tol = 1e-12 * (1.0 + log2((double)N));
    int pass = rel < tol;
    printf("  N=%-6zu  %zu stages  rel=%.2e  tol=%.0e  %s\n",
           N, plan->nstages, rel, tol, pass ? "PASS" : "FAIL");

    vfft_plan_destroy(plan);
    vfft_aligned_free(ir);vfft_aligned_free(ii_);vfft_aligned_free(gr);vfft_aligned_free(gi);vfft_aligned_free(nr);vfft_aligned_free(ni);
    return pass;
}

static int test_roundtrip(size_t N, const vfft_codelet_registry *reg) {
    double *ir = aa64(N), *ii_ = aa64(N);
    double *fr = aa64(N), *fi = aa64(N);
    double *br = aa64(N), *bi = aa64(N);
    fill_rand(ir, N, 3000+(unsigned)N);
    fill_rand(ii_, N, 4000+(unsigned)N);

    vfft_plan *plan = vfft_plan_create(N, reg);
    if (!plan) return 0;

    vfft_execute_fwd(plan, ir, ii_, fr, fi);
    vfft_execute_bwd(plan, fr, fi, br, bi);

    double err = 0, mag = 0;
    for (size_t i = 0; i < N; i++) {
        br[i] /= (double)N; bi[i] /= (double)N;
        double e = fmax(fabs(ir[i]-br[i]), fabs(ii_[i]-bi[i]));
        if (e > err) err = e;
        double m = fmax(fabs(ir[i]), fabs(ii_[i]));
        if (m > mag) mag = m;
    }
    double rel = mag > 0 ? err/mag : err;
    double tol = 1e-12 * (1.0 + log2((double)N));
    int pass = rel < tol;
    printf("  N=%-6zu  roundtrip  rel=%.2e  %s\n", N, rel, pass ? "PASS" : "FAIL");

    vfft_plan_destroy(plan);
    vfft_aligned_free(ir);vfft_aligned_free(ii_);vfft_aligned_free(fr);vfft_aligned_free(fi);vfft_aligned_free(br);vfft_aligned_free(bi);
    return pass;
}

int main(void) {
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  VectorFFT Planner — multi-radix FFT execution engine\n");
    printf("  Using naive codelets (no SIMD optimization)\n");
    printf("════════════════════════════════════════════════════════════════\n\n");

    vfft_codelet_registry reg;
    vfft_registry_init_naive(&reg);

    int p = 0, t = 0;

    /* ── Factorization display ── */
    printf("── Factorization plans ──\n");
    size_t display_Ns[] = {1, 2, 4, 8, 16, 32, 64, 128,
                            6, 10, 12, 15, 24,
                            11, 13, 17, 19, 23,
                            256, 512, 1024,
                            11*64, 13*32, 17*128, 19*16, 23*8,
                            2*3*5*7, 3*5*7*11, 6528};
    for (size_t i = 0; i < sizeof(display_Ns)/sizeof(display_Ns[0]); i++) {
        vfft_plan *plan = vfft_plan_create(display_Ns[i], &reg);
        if (plan) { vfft_plan_print(plan); vfft_plan_destroy(plan); }
    }

    /* ── Powers of 2 ── */
    printf("\n── Powers of 2 ──\n");
    for (int e = 0; e <= 10; e++) {
        t++; p += test_fwd(1u << e, &reg, 0);
    }

    /* ── Small composites ── */
    printf("\n── Small composites ──\n");
    size_t composites[] = {6, 10, 12, 15, 18, 20, 24, 30, 36, 48, 60, 72, 120};
    for (size_t i = 0; i < sizeof(composites)/sizeof(composites[0]); i++) {
        t++; p += test_fwd(composites[i], &reg, 0);
    }

    /* ── Primes (single-stage) ── */
    printf("\n── Primes (single-stage N1 codelet) ──\n");
    size_t primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23};
    for (size_t i = 0; i < sizeof(primes)/sizeof(primes[0]); i++) {
        t++; p += test_fwd(primes[i], &reg, 0);
    }

    /* ── Mixed prime × power-of-2 (multi-stage) ── */
    printf("\n── Prime × power-of-2 ──\n");
    size_t mixed[] = {11*8, 11*64, 13*16, 13*128, 17*8, 17*64, 19*32, 23*16};
    for (size_t i = 0; i < sizeof(mixed)/sizeof(mixed[0]); i++) {
        t++; p += test_fwd(mixed[i], &reg, 0);
    }

    /* ── Multi-prime composites ── */
    printf("\n── Multi-prime composites ──\n");
    size_t multi[] = {3*5*7, 2*3*5*7, 11*13, 11*13*2, 17*3*8, 19*5*4, 23*11*2};
    for (size_t i = 0; i < sizeof(multi)/sizeof(multi[0]); i++) {
        t++; p += test_fwd(multi[i], &reg, 0);
    }

    /* ── Roundtrips ── */
    printf("\n── Roundtrips ──\n");
    size_t rt_Ns[] = {1, 8, 64, 128, 11*8, 13*32, 17*64, 2*3*5*7, 1024};
    for (size_t i = 0; i < sizeof(rt_Ns)/sizeof(rt_Ns[0]); i++) {
        t++; p += test_roundtrip(rt_Ns[i], &reg);
    }

    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("  %d/%d %s\n", p, t, p == t ? "ALL PASSED" : "FAILURES");
    printf("════════════════════════════════════════════════════════════════\n");

    return p != t;
}