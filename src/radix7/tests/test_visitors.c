/**
 * @file  test_visitors.c
 * @brief End-to-end tests for radix-7 stage visitors
 *
 * Simulates what the centralized planner does:
 *   1. Precompute base-7 digit-reversal table
 *   2. Precompute BLOCKED3 twiddle tables per stage
 *   3. Forward:  digit-reverse → stages 0..L-1 via fft_radix7_visit_forward
 *   4. Backward: stages L-1..0 via fft_radix7_visit_backward → digit-reverse
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fft_r7_platform.h"
#include "fft_radix7.h"

/* ── Helpers ─────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;

static inline double now_sec(void)
{
    return r7_now_sec();
}

static void fill_random(double *re, double *im, int N, unsigned seed)
{
    for (int i = 0; i < N; i++)
    {
        seed = seed * 1103515245u + 12345u;
        re[i] = ((double)(seed >> 16) / 32768.0) - 1.0;
        seed = seed * 1103515245u + 12345u;
        im[i] = ((double)(seed >> 16) / 32768.0) - 1.0;
    }
}

static void check(const char *name, double err, double tol)
{
    if (err <= tol)
    {
        g_pass++;
        printf("  PASS %-50s err=%.2e\n", name, err);
    }
    else
    {
        g_fail++;
        printf("  FAIL %-50s err=%.2e > %.2e\n", name, err, tol);
    }
}

static int pow7(int L)
{
    int N = 1;
    for (int i = 0; i < L; i++)
        N *= 7;
    return N;
}

/* ── Digit reversal (planner responsibility) ─────────────────────── */

static void compute_perm(int *perm, int L, int N)
{
    for (int i = 0; i < N; i++)
    {
        int rev = 0, x = i;
        for (int d = 0; d < L; d++)
        {
            rev = rev * 7 + (x % 7);
            x /= 7;
        }
        perm[i] = rev;
    }
}

static void apply_perm(double *re, double *im, const int *perm, int N)
{
    for (int i = 0; i < N; i++)
    {
        int j = perm[i];
        if (j > i)
        {
            double t;
            t = re[i];
            re[i] = re[j];
            re[j] = t;
            t = im[i];
            im[i] = im[j];
            im[j] = t;
        }
    }
}

/* ── Twiddle table (planner responsibility) ──────────────────────── */

typedef struct
{
    double *tw1_re, *tw1_im;
    double *tw2_re, *tw2_im;
    double *tw3_re, *tw3_im;
    int K;
} tw_table_t;

static tw_table_t *alloc_twiddles(int L)
{
    tw_table_t *tw = (tw_table_t *)calloc(L, sizeof(tw_table_t));
    for (int s = 0; s < L; s++)
    {
        int K = pow7(s);
        tw[s].K = K;
        if (s == 0)
            continue; /* stage 0 = N1, no twiddles */

        size_t bytes = ((size_t)K * sizeof(double) + 63) & ~(size_t)63;
        if (bytes < 64)
            bytes = 64;

        tw[s].tw1_re = (double *)r7_aligned_alloc(64, bytes);
        tw[s].tw1_im = (double *)r7_aligned_alloc(64, bytes);
        tw[s].tw2_re = (double *)r7_aligned_alloc(64, bytes);
        tw[s].tw2_im = (double *)r7_aligned_alloc(64, bytes);
        tw[s].tw3_re = (double *)r7_aligned_alloc(64, bytes);
        tw[s].tw3_im = (double *)r7_aligned_alloc(64, bytes);

        double inv_M = 1.0 / (7.0 * K);
        for (int k = 0; k < K; k++)
        {
            double angle = -2.0 * M_PI * k * inv_M;
            tw[s].tw1_re[k] = cos(1.0 * angle);
            tw[s].tw1_im[k] = sin(1.0 * angle);
            tw[s].tw2_re[k] = cos(2.0 * angle);
            tw[s].tw2_im[k] = sin(2.0 * angle);
            tw[s].tw3_re[k] = cos(3.0 * angle);
            tw[s].tw3_im[k] = sin(3.0 * angle);
        }
        /* Zero-pad for SIMD overread */
        int alloc_n = (int)(bytes / sizeof(double));
        for (int k = K; k < alloc_n; k++)
        {
            tw[s].tw1_re[k] = 0;
            tw[s].tw1_im[k] = 0;
            tw[s].tw2_re[k] = 0;
            tw[s].tw2_im[k] = 0;
            tw[s].tw3_re[k] = 0;
            tw[s].tw3_im[k] = 0;
        }
    }
    return tw;
}

static void free_twiddles(tw_table_t *tw, int L)
{
    for (int s = 1; s < L; s++)
    {
        r7_aligned_free(tw[s].tw1_re);
        r7_aligned_free(tw[s].tw1_im);
        r7_aligned_free(tw[s].tw2_re);
        r7_aligned_free(tw[s].tw2_im);
        r7_aligned_free(tw[s].tw3_re);
        r7_aligned_free(tw[s].tw3_im);
    }
    free(tw);
}

/* ── Forward/backward using visitor API ──────────────────────────── */

static void do_forward(const fft_r7_vtable_t *vt, double *re, double *im,
                       int L, int N, const int *perm, const tw_table_t *tw)
{
    apply_perm(re, im, perm, N);
    for (int s = 0; s < L; s++)
    {
        int K = tw[s].K;
        int num_groups = N / (7 * K);
        if (s == 0)
            fft_radix7_visit_forward(vt, re, im, K, num_groups,
                                     NULL, NULL, NULL, NULL, NULL, NULL);
        else
            fft_radix7_visit_forward(vt, re, im, K, num_groups,
                                     tw[s].tw1_re, tw[s].tw1_im,
                                     tw[s].tw2_re, tw[s].tw2_im,
                                     tw[s].tw3_re, tw[s].tw3_im);
    }
}

static void do_backward(const fft_r7_vtable_t *vt, double *re, double *im,
                        int L, int N, const int *perm, const tw_table_t *tw)
{
    for (int s = L - 1; s >= 0; s--)
    {
        int K = tw[s].K;
        int num_groups = N / (7 * K);
        if (s == 0)
            fft_radix7_visit_backward(vt, re, im, K, num_groups,
                                      NULL, NULL, NULL, NULL, NULL, NULL);
        else
            fft_radix7_visit_backward(vt, re, im, K, num_groups,
                                      tw[s].tw1_re, tw[s].tw1_im,
                                      tw[s].tw2_re, tw[s].tw2_im,
                                      tw[s].tw3_re, tw[s].tw3_im);
    }
    apply_perm(re, im, perm, N);
}

/* ── Naive O(N²) DFT reference ───────────────────────────────────── */

static void naive_dft(const double *xr, const double *xi,
                      double *Xr, double *Xi, int N, int sign)
{
    for (int k = 0; k < N; k++)
    {
        double sr = 0.0, si = 0.0;
        for (int n = 0; n < N; n++)
        {
            double angle = sign * 2.0 * M_PI * (double)k * (double)n / (double)N;
            double wr = cos(angle), wi = sin(angle);
            sr += xr[n] * wr - xi[n] * wi;
            si += xr[n] * wi + xi[n] * wr;
        }
        Xr[k] = sr;
        Xi[k] = si;
    }
}

/* ── Test 1: Roundtrip ───────────────────────────────────────────── */

static void test_roundtrip(const fft_r7_vtable_t *vt, int L)
{
    int N = pow7(L);
    double *re = (double *)r7_aligned_alloc(64, N * 8);
    double *im = (double *)r7_aligned_alloc(64, N * 8);
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
    for (int i = 0; i < N; i++)
    {
        double er = fabs(re[i] * inv_N - or_[i]);
        double ei = fabs(im[i] * inv_N - oi_[i]);
        if (er > mx)
            mx = er;
        if (ei > mx)
            mx = ei;
    }

    char label[80];
    snprintf(label, sizeof(label), "roundtrip L=%d  N=%d", L, N);
    check(label, mx, L * 3.0e-15);

    free_twiddles(tw, L);
    free(perm);
    r7_aligned_free(re);
    r7_aligned_free(im);
    free(or_);
    free(oi_);
}

/* ── Test 2: Parseval ────────────────────────────────────────────── */

static void test_parseval(const fft_r7_vtable_t *vt, int L)
{
    int N = pow7(L);
    double *re = (double *)r7_aligned_alloc(64, N * 8);
    double *im = (double *)r7_aligned_alloc(64, N * 8);

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

    free_twiddles(tw, L);
    free(perm);
    r7_aligned_free(re);
    r7_aligned_free(im);
}

/* ── Test 3: Impulse → flat spectrum ─────────────────────────────── */

static void test_impulse(const fft_r7_vtable_t *vt, int L)
{
    int N = pow7(L);
    double *re = (double *)r7_aligned_alloc(64, N * 8);
    double *im = (double *)r7_aligned_alloc(64, N * 8);

    memset(re, 0, N * 8);
    memset(im, 0, N * 8);
    re[0] = 1.0;

    int *perm = (int *)malloc(N * sizeof(int));
    compute_perm(perm, L, N);
    tw_table_t *tw = alloc_twiddles(L);

    do_forward(vt, re, im, L, N, perm, tw);

    double mx = 0.0;
    for (int i = 0; i < N; i++)
    {
        double er = fabs(re[i] - 1.0), ei = fabs(im[i]);
        if (er > mx)
            mx = er;
        if (ei > mx)
            mx = ei;
    }

    char label[80];
    snprintf(label, sizeof(label), "impulse   L=%d  N=%d", L, N);
    check(label, mx, L * 2.0e-15);

    free_twiddles(tw, L);
    free(perm);
    r7_aligned_free(re);
    r7_aligned_free(im);
}

/* ── Test 4: Naive cross-check ───────────────────────────────────── */

static void test_naive(const fft_r7_vtable_t *vt, int L)
{
    int N = pow7(L);
    if (N > 2401)
        return;

    double *re = (double *)r7_aligned_alloc(64, N * 8);
    double *im = (double *)r7_aligned_alloc(64, N * 8);
    double *Rre = (double *)r7_aligned_alloc(64, N * 8);
    double *Rim = (double *)r7_aligned_alloc(64, N * 8);
    double *Nre = (double *)malloc(N * 8);
    double *Nim = (double *)malloc(N * 8);

    fill_random(re, im, N, 8000 + L);
    memcpy(Rre, re, N * 8);
    memcpy(Rim, im, N * 8);

    naive_dft(re, im, Nre, Nim, N, -1);

    int *perm = (int *)malloc(N * sizeof(int));
    compute_perm(perm, L, N);
    tw_table_t *tw = alloc_twiddles(L);

    do_forward(vt, Rre, Rim, L, N, perm, tw);

    double mx = 0.0, max_mag = 0.0;
    for (int i = 0; i < N; i++)
    {
        double er = fabs(Rre[i] - Nre[i]), ei = fabs(Rim[i] - Nim[i]);
        if (er > mx)
            mx = er;
        if (ei > mx)
            mx = ei;
        double mag = sqrt(Nre[i] * Nre[i] + Nim[i] * Nim[i]);
        if (mag > max_mag)
            max_mag = mag;
    }
    double rel = (max_mag > 0.0) ? mx / max_mag : mx;

    char label[80];
    snprintf(label, sizeof(label), "naive     L=%d  N=%d (rel)", L, N);
    check(label, rel, L * L * 5.0e-14);

    free_twiddles(tw, L);
    free(perm);
    r7_aligned_free(re);
    r7_aligned_free(im);
    r7_aligned_free(Rre);
    r7_aligned_free(Rim);
    free(Nre);
    free(Nim);
}

/* ── Benchmark ───────────────────────────────────────────────────── */

static void bench_fft(const fft_r7_vtable_t *vt, int L)
{
    int N = pow7(L);
    double *re = (double *)r7_aligned_alloc(64, N * 8);
    double *im = (double *)r7_aligned_alloc(64, N * 8);
    fill_random(re, im, N, 42 + L);

    int *perm = (int *)malloc(N * sizeof(int));
    compute_perm(perm, L, N);
    tw_table_t *tw = alloc_twiddles(L);

    /* Warmup */
    do_forward(vt, re, im, L, N, perm, tw);
    fill_random(re, im, N, 42 + L);

    long iters = 0;
    double t0 = now_sec(), elapsed;
    do
    {
        do_forward(vt, re, im, L, N, perm, tw);
        iters++;
        elapsed = now_sec() - t0;
    } while (elapsed < 0.15);

    double flops = 5.0 * N * L;
    double gflops = (double)iters * flops / elapsed / 1e9;
    double us = elapsed / iters * 1e6;
    double ns_pt = elapsed / iters / N * 1e9;

    printf("  L=%d  N=%7d  %8.2f GFLOP/s  %8.1f us/FFT  %5.1f ns/pt  "
           "(%ld iters, %s)\n",
           L, N, gflops, us, ns_pt, iters, fft_r7_isa_name(vt->isa));

    free_twiddles(tw, L);
    free(perm);
    r7_aligned_free(re);
    r7_aligned_free(im);
}

/* ── Main ────────────────────────────────────────────────────────── */

int main(void)
{
    fft_r7_vtable_t vt;
    fft_radix7_init_vtable(&vt);

    printf("========================================================\n");
    printf("  Radix-7 Visitor End-to-End Tests — ISA: %s\n",
           fft_r7_isa_name(vt.isa));
    printf("========================================================\n\n");

    int max_L = 6;

    printf("--- Roundtrip ---\n");
    for (int L = 1; L <= max_L; L++)
        test_roundtrip(&vt, L);

    printf("\n--- Parseval ---\n");
    for (int L = 1; L <= max_L; L++)
        test_parseval(&vt, L);

    printf("\n--- Impulse ---\n");
    for (int L = 1; L <= max_L; L++)
        test_impulse(&vt, L);

    printf("\n--- Naive Cross-Check ---\n");
    for (int L = 1; L <= 4; L++)
        test_naive(&vt, L);

    printf("\n--- Benchmark ---\n");
    for (int L = 1; L <= max_L; L++)
        bench_fft(&vt, L);

    printf("\n========================================================\n");
    printf("Results: %d PASS, %d FAIL\n", g_pass, g_fail);
    printf("========================================================\n");

    return g_fail ? 1 : 0;
}