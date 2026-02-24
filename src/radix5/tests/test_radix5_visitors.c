/**
 * @file  test_radix5_visitors.c
 * @brief End-to-end tests and benchmark for radix-5 stage visitors
 *
 * Simulates what the centralized planner does:
 *   1. Precompute base-5 digit-reversal table
 *   2. Precompute BLOCKED2 twiddle tables per stage
 *   3. Forward:  digit-reverse → stages 0..L-1 via fft_radix5_visit_forward
 *   4. Backward: stages L-1..0 via fft_radix5_visit_backward → digit-reverse
 *
 * Tests:
 *   - Roundtrip (fwd → bwd → /N recovers original)
 *   - Parseval  (energy conservation)
 *   - Impulse   (δ → flat spectrum)
 *   - Naive     (O(N²) cross-check)
 *   - Linearity (α·x + β·y transforms linearly)
 *   - Shift     (circular shift ↔ phase ramp)
 *
 * Benchmark:
 *   - Auto-timed throughput for L = 1..8 (N = 5..390625)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fft_r5_platform.h"
#include "fft_radix5.h"

/* ================================================================== */
/*  Helpers                                                            */
/* ================================================================== */

static int g_pass = 0, g_fail = 0;

static inline double now_sec(void) { return r5_now_sec(); }

static void fill_random(double *re, double *im, int N, unsigned seed) {
    for (int i = 0; i < N; i++) {
        seed = seed * 1103515245u + 12345u;
        re[i] = ((double)(seed >> 16) / 32768.0) - 1.0;
        seed = seed * 1103515245u + 12345u;
        im[i] = ((double)(seed >> 16) / 32768.0) - 1.0;
    }
}

static void check(const char *name, double err, double tol) {
    if (err <= tol) {
        g_pass++;
        printf("  PASS %-50s err=%.2e\n", name, err);
    } else {
        g_fail++;
        printf("  FAIL %-50s err=%.2e > %.2e\n", name, err, tol);
    }
}

static int pow5(int L) {
    int N = 1;
    for (int i = 0; i < L; i++) N *= 5;
    return N;
}

/* ================================================================== */
/*  Digit reversal (planner responsibility)                            */
/* ================================================================== */

static void compute_perm(int *perm, int L, int N) {
    for (int i = 0; i < N; i++) {
        int rev = 0, x = i;
        for (int d = 0; d < L; d++) {
            rev = rev * 5 + (x % 5);
            x /= 5;
        }
        perm[i] = rev;
    }
}

static void apply_perm(double *re, double *im, const int *perm, int N) {
    for (int i = 0; i < N; i++) {
        int j = perm[i];
        if (j > i) {
            double t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
}

/* ================================================================== */
/*  BLOCKED2 twiddle tables (planner responsibility)                   */
/*                                                                     */
/*  Stage s, K = 5^s, M = 5·K:                                        */
/*    tw1[k] = exp(-2πi·k / M)     = W^k                             */
/*    tw2[k] = exp(-2πi·2k / M)    = W^{2k}                          */
/*  W3 = W1·W2, W4 = W2² derived in butterfly                         */
/* ================================================================== */

typedef struct {
    double *tw1_re, *tw1_im;
    double *tw2_re, *tw2_im;
    int K;
} tw_table_t;

static tw_table_t *alloc_twiddles(int L) {
    tw_table_t *tw = (tw_table_t *)calloc(L, sizeof(tw_table_t));
    for (int s = 0; s < L; s++) {
        int K = pow5(s);
        tw[s].K = K;
        if (s == 0) continue;   /* stage 0 = N1, no twiddles */

        size_t bytes = ((size_t)K * sizeof(double) + 63) & ~(size_t)63;
        if (bytes < 64) bytes = 64;

        tw[s].tw1_re = (double *)r5_aligned_alloc(64, bytes);
        tw[s].tw1_im = (double *)r5_aligned_alloc(64, bytes);
        tw[s].tw2_re = (double *)r5_aligned_alloc(64, bytes);
        tw[s].tw2_im = (double *)r5_aligned_alloc(64, bytes);

        double inv_M = 1.0 / (5.0 * K);
        for (int k = 0; k < K; k++) {
            double angle = -2.0 * M_PI * k * inv_M;
            tw[s].tw1_re[k] = cos(1.0 * angle);
            tw[s].tw1_im[k] = sin(1.0 * angle);
            tw[s].tw2_re[k] = cos(2.0 * angle);
            tw[s].tw2_im[k] = sin(2.0 * angle);
        }
        /* Zero-pad for SIMD overread */
        int alloc_n = (int)(bytes / sizeof(double));
        for (int k = K; k < alloc_n; k++) {
            tw[s].tw1_re[k] = 0; tw[s].tw1_im[k] = 0;
            tw[s].tw2_re[k] = 0; tw[s].tw2_im[k] = 0;
        }
    }
    return tw;
}

static void free_twiddles(tw_table_t *tw, int L) {
    for (int s = 1; s < L; s++) {
        r5_aligned_free(tw[s].tw1_re); r5_aligned_free(tw[s].tw1_im);
        r5_aligned_free(tw[s].tw2_re); r5_aligned_free(tw[s].tw2_im);
    }
    free(tw);
}

/* ================================================================== */
/*  Forward / backward using visitor API                               */
/* ================================================================== */

static void do_forward(const fft_r5_vtable_t *vt, double *re, double *im,
                       int L, int N, const int *perm, const tw_table_t *tw)
{
    apply_perm(re, im, perm, N);
    for (int s = 0; s < L; s++) {
        int K = tw[s].K;
        int num_groups = N / (5 * K);
        if (s == 0)
            fft_radix5_visit_forward(vt, re, im, K, num_groups,
                                     NULL, NULL, NULL, NULL);
        else
            fft_radix5_visit_forward(vt, re, im, K, num_groups,
                                     tw[s].tw1_re, tw[s].tw1_im,
                                     tw[s].tw2_re, tw[s].tw2_im);
    }
}

static void do_backward(const fft_r5_vtable_t *vt, double *re, double *im,
                        int L, int N, const int *perm, const tw_table_t *tw)
{
    for (int s = L - 1; s >= 0; s--) {
        int K = tw[s].K;
        int num_groups = N / (5 * K);
        if (s == 0)
            fft_radix5_visit_backward(vt, re, im, K, num_groups,
                                      NULL, NULL, NULL, NULL);
        else
            fft_radix5_visit_backward(vt, re, im, K, num_groups,
                                      tw[s].tw1_re, tw[s].tw1_im,
                                      tw[s].tw2_re, tw[s].tw2_im);
    }
    apply_perm(re, im, perm, N);
}

/* ================================================================== */
/*  Naive O(N²) DFT reference                                         */
/* ================================================================== */

static void naive_dft(const double *xr, const double *xi,
                      double *Xr, double *Xi, int N, int sign)
{
    for (int k = 0; k < N; k++) {
        double sr = 0.0, si = 0.0;
        for (int n = 0; n < N; n++) {
            double angle = sign * 2.0 * M_PI * (double)k * (double)n / (double)N;
            double wr = cos(angle), wi = sin(angle);
            sr += xr[n] * wr - xi[n] * wi;
            si += xr[n] * wi + xi[n] * wr;
        }
        Xr[k] = sr;
        Xi[k] = si;
    }
}

/* ================================================================== */
/*  Test 1: Roundtrip                                                  */
/* ================================================================== */

static void test_roundtrip(const fft_r5_vtable_t *vt, int L) {
    int N = pow5(L);
    double *re  = (double *)r5_aligned_alloc(64, N * 8);
    double *im  = (double *)r5_aligned_alloc(64, N * 8);
    double *or_ = (double *)malloc(N * 8);
    double *oi_ = (double *)malloc(N * 8);

    fill_random(re, im, N, 1000 + L);
    memcpy(or_, re, N * 8);
    memcpy(oi_, im, N * 8);

    int *perm = (int *)malloc(N * sizeof(int));
    compute_perm(perm, L, N);
    tw_table_t *tw = alloc_twiddles(L);

    do_forward(vt, re, im, L, N, perm, tw);
    do_backward(vt, re, im, L, N, perm, tw);

    double inv_N = 1.0 / (double)N;
    double mx = 0.0;
    for (int i = 0; i < N; i++) {
        double er = fabs(re[i] * inv_N - or_[i]);
        double ei = fabs(im[i] * inv_N - oi_[i]);
        if (er > mx) mx = er;
        if (ei > mx) mx = ei;
    }

    char label[80];
    snprintf(label, sizeof(label), "roundtrip L=%d  N=%d", L, N);
    check(label, mx, L * 3.0e-15);

    free_twiddles(tw, L); free(perm);
    r5_aligned_free(re); r5_aligned_free(im); free(or_); free(oi_);
}

/* ================================================================== */
/*  Test 2: Parseval (energy conservation)                             */
/* ================================================================== */

static void test_parseval(const fft_r5_vtable_t *vt, int L) {
    int N = pow5(L);
    double *re = (double *)r5_aligned_alloc(64, N * 8);
    double *im = (double *)r5_aligned_alloc(64, N * 8);

    fill_random(re, im, N, 2000 + L);

    double E_time = 0.0;
    for (int i = 0; i < N; i++)
        E_time += re[i] * re[i] + im[i] * im[i];

    int *perm = (int *)malloc(N * sizeof(int));
    compute_perm(perm, L, N);
    tw_table_t *tw = alloc_twiddles(L);

    do_forward(vt, re, im, L, N, perm, tw);

    double E_freq = 0.0;
    for (int i = 0; i < N; i++)
        E_freq += re[i] * re[i] + im[i] * im[i];
    E_freq /= (double)N;

    double rel = fabs(E_freq - E_time) / E_time;

    char label[80];
    snprintf(label, sizeof(label), "Parseval  L=%d  N=%d", L, N);
    check(label, rel, L * 5.0e-14);

    free_twiddles(tw, L); free(perm);
    r5_aligned_free(re); r5_aligned_free(im);
}

/* ================================================================== */
/*  Test 3: Impulse → flat spectrum                                    */
/* ================================================================== */

static void test_impulse(const fft_r5_vtable_t *vt, int L) {
    int N = pow5(L);
    double *re = (double *)r5_aligned_alloc(64, N * 8);
    double *im = (double *)r5_aligned_alloc(64, N * 8);

    memset(re, 0, N * 8); memset(im, 0, N * 8);
    re[0] = 1.0;

    int *perm = (int *)malloc(N * sizeof(int));
    compute_perm(perm, L, N);
    tw_table_t *tw = alloc_twiddles(L);

    do_forward(vt, re, im, L, N, perm, tw);

    double mx = 0.0;
    for (int i = 0; i < N; i++) {
        double er = fabs(re[i] - 1.0), ei = fabs(im[i]);
        if (er > mx) mx = er;
        if (ei > mx) mx = ei;
    }

    char label[80];
    snprintf(label, sizeof(label), "impulse   L=%d  N=%d", L, N);
    check(label, mx, L * 2.0e-15);

    free_twiddles(tw, L); free(perm);
    r5_aligned_free(re); r5_aligned_free(im);
}

/* ================================================================== */
/*  Test 4: Naive O(N²) cross-check                                   */
/* ================================================================== */

static void test_naive(const fft_r5_vtable_t *vt, int L) {
    int N = pow5(L);
    if (N > 3125) return;   /* skip for L≥6, O(N²) too slow */

    double *re  = (double *)r5_aligned_alloc(64, N * 8);
    double *im  = (double *)r5_aligned_alloc(64, N * 8);
    double *Rre = (double *)r5_aligned_alloc(64, N * 8);
    double *Rim = (double *)r5_aligned_alloc(64, N * 8);
    double *Nre = (double *)malloc(N * 8);
    double *Nim = (double *)malloc(N * 8);

    fill_random(re, im, N, 8000 + L);
    memcpy(Rre, re, N * 8); memcpy(Rim, im, N * 8);

    naive_dft(re, im, Nre, Nim, N, -1);

    int *perm = (int *)malloc(N * sizeof(int));
    compute_perm(perm, L, N);
    tw_table_t *tw = alloc_twiddles(L);

    do_forward(vt, Rre, Rim, L, N, perm, tw);

    double mx = 0.0, max_mag = 0.0;
    for (int i = 0; i < N; i++) {
        double er = fabs(Rre[i] - Nre[i]), ei = fabs(Rim[i] - Nim[i]);
        if (er > mx) mx = er;
        if (ei > mx) mx = ei;
        double mag = sqrt(Nre[i] * Nre[i] + Nim[i] * Nim[i]);
        if (mag > max_mag) max_mag = mag;
    }
    double rel = (max_mag > 0.0) ? mx / max_mag : mx;

    char label[80];
    snprintf(label, sizeof(label), "naive     L=%d  N=%d (rel)", L, N);
    check(label, rel, L * L * 5.0e-14);

    free_twiddles(tw, L); free(perm);
    r5_aligned_free(re); r5_aligned_free(im);
    r5_aligned_free(Rre); r5_aligned_free(Rim);
    free(Nre); free(Nim);
}

/* ================================================================== */
/*  Test 5: Linearity — DFT(α·x + β·y) = α·DFT(x) + β·DFT(y)       */
/* ================================================================== */

static void test_linearity(const fft_r5_vtable_t *vt, int L) {
    int N = pow5(L);
    double alpha = 2.7, beta = -1.3;

    double *xr = (double *)r5_aligned_alloc(64, N * 8);
    double *xi = (double *)r5_aligned_alloc(64, N * 8);
    double *yr = (double *)r5_aligned_alloc(64, N * 8);
    double *yi = (double *)r5_aligned_alloc(64, N * 8);
    double *zr = (double *)r5_aligned_alloc(64, N * 8);
    double *zi = (double *)r5_aligned_alloc(64, N * 8);

    fill_random(xr, xi, N, 3000 + L);
    fill_random(yr, yi, N, 4000 + L);
    for (int i = 0; i < N; i++) {
        zr[i] = alpha * xr[i] + beta * yr[i];
        zi[i] = alpha * xi[i] + beta * yi[i];
    }

    int *perm = (int *)malloc(N * sizeof(int));
    compute_perm(perm, L, N);
    tw_table_t *tw = alloc_twiddles(L);

    do_forward(vt, xr, xi, L, N, perm, tw);
    do_forward(vt, yr, yi, L, N, perm, tw);
    do_forward(vt, zr, zi, L, N, perm, tw);

    double mx = 0.0;
    for (int i = 0; i < N; i++) {
        double er = fabs(zr[i] - (alpha * xr[i] + beta * yr[i]));
        double ei = fabs(zi[i] - (alpha * xi[i] + beta * yi[i]));
        if (er > mx) mx = er;
        if (ei > mx) mx = ei;
    }

    char label[80];
    snprintf(label, sizeof(label), "linearity L=%d  N=%d", L, N);
    check(label, mx, N * L * 1.0e-13);

    free_twiddles(tw, L); free(perm);
    r5_aligned_free(xr); r5_aligned_free(xi);
    r5_aligned_free(yr); r5_aligned_free(yi);
    r5_aligned_free(zr); r5_aligned_free(zi);
}

/* ================================================================== */
/*  Test 6: Circular shift ↔ phase ramp                               */
/*  x[n-d mod N]  ↔  X[k] · exp(-2πi·k·d/N)                        */
/* ================================================================== */

static void test_shift(const fft_r5_vtable_t *vt, int L) {
    int N = pow5(L);
    int d = N / 3 + 1;   /* arbitrary non-trivial shift */

    double *xr = (double *)r5_aligned_alloc(64, N * 8);
    double *xi = (double *)r5_aligned_alloc(64, N * 8);
    double *sr = (double *)r5_aligned_alloc(64, N * 8);
    double *si = (double *)r5_aligned_alloc(64, N * 8);

    fill_random(xr, xi, N, 5000 + L);
    /* Circularly shift x by d positions */
    for (int i = 0; i < N; i++) {
        int j = (i + d) % N;
        sr[i] = xr[j];
        si[i] = xi[j];
    }

    int *perm = (int *)malloc(N * sizeof(int));
    compute_perm(perm, L, N);
    tw_table_t *tw = alloc_twiddles(L);

    /* Transform both */
    do_forward(vt, xr, xi, L, N, perm, tw);
    do_forward(vt, sr, si, L, N, perm, tw);

    /* shifted_X[k] should equal X[k] * exp(-2πi·k·d/N) */
    double mx = 0.0;
    for (int k = 0; k < N; k++) {
        double angle = +2.0 * M_PI * k * d / N;
        double wr = cos(angle), wi = sin(angle);
        double expect_r = xr[k] * wr - xi[k] * wi;
        double expect_i = xr[k] * wi + xi[k] * wr;
        double er = fabs(sr[k] - expect_r);
        double ei = fabs(si[k] - expect_i);
        if (er > mx) mx = er;
        if (ei > mx) mx = ei;
    }

    char label[80];
    snprintf(label, sizeof(label), "shift     L=%d  N=%d  d=%d", L, N, d);
    check(label, mx, N * L * 1.0e-13);

    free_twiddles(tw, L); free(perm);
    r5_aligned_free(xr); r5_aligned_free(xi);
    r5_aligned_free(sr); r5_aligned_free(si);
}

/* ================================================================== */
/*  Benchmark                                                          */
/* ================================================================== */

static void bench_fft(const fft_r5_vtable_t *vt, int L) {
    int N = pow5(L);
    double *re = (double *)r5_aligned_alloc(64, N * 8);
    double *im = (double *)r5_aligned_alloc(64, N * 8);
    fill_random(re, im, N, 42 + L);

    int *perm = (int *)malloc(N * sizeof(int));
    compute_perm(perm, L, N);
    tw_table_t *tw = alloc_twiddles(L);

    /* Warmup */
    do_forward(vt, re, im, L, N, perm, tw);
    fill_random(re, im, N, 42 + L);

    /* Auto-time: at least 0.2s or 50 iters */
    long iters = 0;
    double t0 = now_sec(), elapsed;
    do {
        do_forward(vt, re, im, L, N, perm, tw);
        iters++;
        elapsed = now_sec() - t0;
    } while (elapsed < 0.2 && iters < 500000);

    /*
     * Flop count for radix-5 WFTA butterfly:
     *   6 real muls + 17 real adds = 23 flops per butterfly
     *   N/5 butterflies per stage, L stages
     *   Twiddle: 4 cmuls per butterfly = 24 flops (stages 1..L-1)
     *
     * Simplified: ~5·N·log₅(N) ≈ 5·N·L as rough GFLOP numerator
     * (matches the standard 5·N·log_r(N) convention)
     */
    double flops = 5.0 * N * L;
    double gflops = (double)iters * flops / elapsed / 1e9;
    double us     = elapsed / iters * 1e6;
    double ns_pt  = elapsed / iters / N * 1e9;

    printf("  L=%d  N=%7d  %8.2f GFLOP/s  %8.1f us/FFT  %5.1f ns/pt  "
           "(%ld iters)\n",
           L, N, gflops, us, ns_pt, iters);

    free_twiddles(tw, L); free(perm);
    r5_aligned_free(re); r5_aligned_free(im);
}

static void bench_roundtrip(const fft_r5_vtable_t *vt, int L) {
    int N = pow5(L);
    double *re = (double *)r5_aligned_alloc(64, N * 8);
    double *im = (double *)r5_aligned_alloc(64, N * 8);
    fill_random(re, im, N, 99 + L);

    int *perm = (int *)malloc(N * sizeof(int));
    compute_perm(perm, L, N);
    tw_table_t *tw = alloc_twiddles(L);

    /* Warmup */
    do_forward(vt, re, im, L, N, perm, tw);
    do_backward(vt, re, im, L, N, perm, tw);
    fill_random(re, im, N, 99 + L);

    long iters = 0;
    double t0 = now_sec(), elapsed;
    do {
        do_forward(vt, re, im, L, N, perm, tw);
        do_backward(vt, re, im, L, N, perm, tw);
        iters++;
        elapsed = now_sec() - t0;
    } while (elapsed < 0.2 && iters < 500000);

    double flops = 2.0 * 5.0 * N * L;   /* fwd + bwd */
    double gflops = (double)iters * flops / elapsed / 1e9;
    double us     = elapsed / iters * 1e6;
    double ns_pt  = elapsed / iters / N * 1e9;

    printf("  L=%d  N=%7d  %8.2f GFLOP/s  %8.1f us/fwd+bwd  %5.1f ns/pt  "
           "(%ld iters)\n",
           L, N, gflops, us, ns_pt, iters);

    free_twiddles(tw, L); free(perm);
    r5_aligned_free(re); r5_aligned_free(im);
}

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main(void) {
    fft_r5_vtable_t vt;
    fft_radix5_init_vtable(&vt);

    printf("========================================================\n");
    printf("  Radix-5 WFTA Visitor End-to-End Tests — ISA: %s\n",
           fft_r5_isa_name(vt.isa));
    printf("========================================================\n\n");

    int max_L = 8;   /* N up to 5^8 = 390625 */

    printf("--- Roundtrip (fwd → bwd → /N) ---\n");
    for (int L = 1; L <= max_L; L++) test_roundtrip(&vt, L);

    printf("\n--- Parseval (energy conservation) ---\n");
    for (int L = 1; L <= max_L; L++) test_parseval(&vt, L);

    printf("\n--- Impulse (δ → flat spectrum) ---\n");
    for (int L = 1; L <= max_L; L++) test_impulse(&vt, L);

    printf("\n--- Naive O(N²) Cross-Check ---\n");
    for (int L = 1; L <= 5; L++) test_naive(&vt, L);  /* up to N=3125 */

    printf("\n--- Linearity (αx + βy) ---\n");
    for (int L = 1; L <= max_L; L++) test_linearity(&vt, L);

    printf("\n--- Shift (circular shift ↔ phase ramp) ---\n");
    for (int L = 1; L <= max_L; L++) test_shift(&vt, L);

    printf("\n--- Forward Benchmark ---\n");
    for (int L = 1; L <= max_L; L++) bench_fft(&vt, L);

    printf("\n--- Roundtrip Benchmark (fwd + bwd) ---\n");
    for (int L = 1; L <= max_L; L++) bench_roundtrip(&vt, L);

    printf("\n========================================================\n");
    printf("Results: %d PASS, %d FAIL\n", g_pass, g_fail);
    printf("========================================================\n");

    return g_fail ? 1 : 0;
}
