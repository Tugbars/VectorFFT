/**
 * @file test_roundtrip.c
 * @brief Round-trip test: forward → backward → verify input recovery
 *
 * For a single radix-4 DIF stage, each k-group of 4 elements at positions
 * {k, K+k, 2K+k, 3K+k} is an independent 4-point butterfly. The round-trip
 * property is:
 *
 *   bv_n1(fv_n1(x))[i] = 4 * x[i]   for all i
 *
 * The scale factor is 4 (the radix), NOT N=4K.
 *
 * For twiddle stages with W_fwd and W_bwd = conj(W_fwd):
 *   bv(fv(x, W_fwd), W_bwd)[i] = 4 * x[i]
 *
 * Build:
 *   gcc -mavx2 -mfma -O2 -o test_roundtrip test_roundtrip.c fft_radix4_fv.c fft_radix4_bv.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "fft_radix4.h"

extern void fft_radix4_fv(double *restrict, double *restrict,
    const double *restrict, const double *restrict,
    const fft_twiddles_soa *restrict, int);
extern void fft_radix4_fv_n1(double *restrict, double *restrict,
    const double *restrict, const double *restrict, int);
extern void fft_radix4_bv(double *restrict, double *restrict,
    const double *restrict, const double *restrict,
    const fft_twiddles_soa *restrict, int);
extern void fft_radix4_bv_n1(double *restrict, double *restrict,
    const double *restrict, const double *restrict, int);

static double *alloc_a(size_t n)
{
    double *p = NULL;
    if (posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }
    return p;
}

static int test_n1(int K)
{
    int N = 4 * K;
    double *x_re  = alloc_a(N), *x_im  = alloc_a(N);
    double *fwd_re = alloc_a(N), *fwd_im = alloc_a(N);
    double *rt_re  = alloc_a(N), *rt_im  = alloc_a(N);

    for (int i = 0; i < N; i++) {
        x_re[i] = sin(0.7 * i + 0.1);
        x_im[i] = cos(0.3 * i + 0.5);
    }

    fft_radix4_fv_n1(fwd_re, fwd_im, x_re, x_im, K);
    fft_radix4_bv_n1(rt_re, rt_im, fwd_re, fwd_im, K);

    double max_err = 0;
    for (int i = 0; i < N; i++) {
        double er = fabs(rt_re[i] / 4.0 - x_re[i]);
        double ei = fabs(rt_im[i] / 4.0 - x_im[i]);
        if (er > max_err) max_err = er;
        if (ei > max_err) max_err = ei;
    }

    bool pass = max_err < 1e-12;
    printf("  n1       K=%5d  N=%6d  max_err=%.2e  %s\n", K, N, max_err, pass ? "PASS" : "FAIL");
    free(x_re); free(x_im); free(fwd_re); free(fwd_im); free(rt_re); free(rt_im);
    return pass ? 0 : 1;
}

static int __attribute__((unused)) test_tw(int K)
{
    int N = 4 * K;
    double *x_re    = alloc_a(N), *x_im    = alloc_a(N);
    double *fwd_re   = alloc_a(N), *fwd_im   = alloc_a(N);
    double *rt_re    = alloc_a(N), *rt_im    = alloc_a(N);
    double *twf_re = alloc_a(3*K), *twf_im = alloc_a(3*K);
    double *twb_re = alloc_a(3*K), *twb_im = alloc_a(3*K);

    for (int i = 0; i < N; i++) {
        x_re[i] = sin(0.7 * i + 0.1);
        x_im[i] = cos(0.3 * i + 0.5);
    }

    for (int k = 0; k < K; k++) {
        double a1 = -2.0 * M_PI * k / N;
        twf_re[k]       = cos(a1);     twf_im[k]       = sin(a1);
        twf_re[K+k]     = cos(2*a1);   twf_im[K+k]     = sin(2*a1);
        twf_re[2*K+k]   = cos(3*a1);   twf_im[2*K+k]   = sin(3*a1);
        twb_re[k]       =  twf_re[k];       twb_im[k]       = -twf_im[k];
        twb_re[K+k]     =  twf_re[K+k];     twb_im[K+k]     = -twf_im[K+k];
        twb_re[2*K+k]   =  twf_re[2*K+k];   twb_im[2*K+k]   = -twf_im[2*K+k];
    }

    fft_twiddles_soa tw_fwd = { .re = twf_re, .im = twf_im };
    fft_twiddles_soa tw_bwd = { .re = twb_re, .im = twb_im };

    fft_radix4_fv(fwd_re, fwd_im, x_re, x_im, &tw_fwd, K);
    fft_radix4_bv(rt_re, rt_im, fwd_re, fwd_im, &tw_bwd, K);

    double max_err = 0;
    for (int i = 0; i < N; i++) {
        double er = fabs(rt_re[i] / 4.0 - x_re[i]);
        double ei = fabs(rt_im[i] / 4.0 - x_im[i]);
        if (er > max_err) max_err = er;
        if (ei > max_err) max_err = ei;
    }

    bool pass = max_err < 1e-12;
    printf("  tw       K=%5d  N=%6d  max_err=%.2e  %s\n", K, N, max_err, pass ? "PASS" : "FAIL");
    free(x_re); free(x_im); free(fwd_re); free(fwd_im); free(rt_re); free(rt_im);
    free(twf_re); free(twf_im); free(twb_re); free(twb_im);
    return pass ? 0 : 1;
}

static int test_mixed(int K)
{
    int N = 4 * K;
    double *x_re  = alloc_a(N), *x_im  = alloc_a(N);
    double *fwd_re = alloc_a(N), *fwd_im = alloc_a(N);
    double *rt_re  = alloc_a(N), *rt_im  = alloc_a(N);
    double *tw_re = alloc_a(3*K), *tw_im = alloc_a(3*K);

    for (int i = 0; i < N; i++) {
        x_re[i] = sin(0.7 * i + 0.1);
        x_im[i] = cos(0.3 * i + 0.5);
    }

    for (int k = 0; k < 3 * K; k++) { tw_re[k] = 1.0; tw_im[k] = 0.0; }
    fft_twiddles_soa tw = { .re = tw_re, .im = tw_im };

    int failures = 0;

    /* n1 forward → tw backward (unity) */
    fft_radix4_fv_n1(fwd_re, fwd_im, x_re, x_im, K);
    fft_radix4_bv(rt_re, rt_im, fwd_re, fwd_im, &tw, K);

    double max_err = 0;
    for (int i = 0; i < N; i++) {
        double er = fabs(rt_re[i]/4.0 - x_re[i]);
        double ei = fabs(rt_im[i]/4.0 - x_im[i]);
        if (er > max_err) max_err = er;
        if (ei > max_err) max_err = ei;
    }
    bool p1 = max_err < 1e-12;
    printf("  n1→tw    K=%5d  N=%6d  max_err=%.2e  %s\n", K, N, max_err, p1 ? "PASS" : "FAIL");
    if (!p1) failures++;

    /* tw forward (unity) → n1 backward */
    fft_radix4_fv(fwd_re, fwd_im, x_re, x_im, &tw, K);
    fft_radix4_bv_n1(rt_re, rt_im, fwd_re, fwd_im, K);

    max_err = 0;
    for (int i = 0; i < N; i++) {
        double er = fabs(rt_re[i]/4.0 - x_re[i]);
        double ei = fabs(rt_im[i]/4.0 - x_im[i]);
        if (er > max_err) max_err = er;
        if (ei > max_err) max_err = ei;
    }
    bool p2 = max_err < 1e-12;
    printf("  tw→n1    K=%5d  N=%6d  max_err=%.2e  %s\n", K, N, max_err, p2 ? "PASS" : "FAIL");
    if (!p2) failures++;

    free(x_re); free(x_im); free(fwd_re); free(fwd_im);
    free(rt_re); free(rt_im); free(tw_re); free(tw_im);
    return failures;
}

static int test_reverse(int K)
{
    int N = 4 * K;
    double *x_re  = alloc_a(N), *x_im  = alloc_a(N);
    double *bwd_re = alloc_a(N), *bwd_im = alloc_a(N);
    double *rt_re  = alloc_a(N), *rt_im  = alloc_a(N);

    for (int i = 0; i < N; i++) {
        x_re[i] = sin(0.7 * i + 0.1);
        x_im[i] = cos(0.3 * i + 0.5);
    }

    fft_radix4_bv_n1(bwd_re, bwd_im, x_re, x_im, K);
    fft_radix4_fv_n1(rt_re, rt_im, bwd_re, bwd_im, K);

    double max_err = 0;
    for (int i = 0; i < N; i++) {
        double er = fabs(rt_re[i]/4.0 - x_re[i]);
        double ei = fabs(rt_im[i]/4.0 - x_im[i]);
        if (er > max_err) max_err = er;
        if (ei > max_err) max_err = ei;
    }
    bool pass = max_err < 1e-12;
    printf("  bv→fv    K=%5d  N=%6d  max_err=%.2e  %s\n", K, N, max_err, pass ? "PASS" : "FAIL");
    free(x_re); free(x_im); free(bwd_re); free(bwd_im); free(rt_re); free(rt_im);
    return pass ? 0 : 1;
}

int main(void)
{
    int Ks[] = {1, 2, 3, 4, 5, 7, 8, 12, 16, 17, 32, 64, 128, 256, 1024, 4096};
    int n = sizeof(Ks) / sizeof(Ks[0]);
    int fail = 0;

    printf("=== Round-trip: bv(fv(x)) = 4*x ===\n\n");

    printf("--- N1 forward → N1 backward ---\n");
    for (int i = 0; i < n; i++) fail += test_n1(Ks[i]);

    printf("\n--- Mixed (n1↔tw, unity twiddles) ---\n");
    for (int i = 0; i < n; i++) fail += test_mixed(Ks[i]);

    printf("\n--- Reverse: N1 backward → N1 forward ---\n");
    for (int i = 0; i < n; i++) fail += test_reverse(Ks[i]);

    printf("\n%s\n", fail == 0 ? "All round-trip tests PASSED" : "SOME TESTS FAILED");
    return fail;
}
