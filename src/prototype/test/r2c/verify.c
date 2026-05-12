/* Verify R=16 r2c codelet against ground-truth DFT.
 *
 * Generate N=16 random real inputs, run codelet, compute reference DFT
 * by direct evaluation, compare X[0..N/2] real and imaginary parts.
 *
 * Calling convention (current emit; same as c2c for now):
 *   void f(in_re, in_im, out_re, out_im, tw_re, tw_im, K)
 * The codelet ignores in_im, tw_re, tw_im. It reads in_re[j*K + k] for
 * j=0..15, k=0..K-1, writes out_re/out_im[k*K + k] for k=0..8.
 *
 * Bench K = 8 (one AVX-512 lane-group). 16 reals × 8 batches = 128 inputs.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <immintrin.h>

__attribute__((target("avx512f,avx512dq,fma")))
void radix16_n1_fwd_avx512_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im,
    size_t K);

#define N 16
#define K_BATCH 8

static double *aa(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(double)) != 0) exit(1);
    return (double *)p;
}

/* Reference: directly compute X[k] = sum_n x[n] * exp(-2*pi*i*n*k/N) */
static void ref_r2c_fwd(const double *x, double *X_re, double *X_im) {
    for (int k = 0; k <= N/2; k++) {
        double re = 0, im = 0;
        for (int n = 0; n < N; n++) {
            double theta = -2.0 * M_PI * (double)(n * k) / (double)N;
            re += x[n] * cos(theta);
            im += x[n] * sin(theta);
        }
        X_re[k] = re;
        X_im[k] = im;
    }
}

int main(void) {
    /* Allocate batched: K_BATCH parallel transforms */
    double *in_re   = aa(N * K_BATCH);
    double *in_im   = aa(N * K_BATCH);    /* unused; codelet doesn't read */
    double *out_re  = aa(N * K_BATCH);    /* slots 0..N/2 written; 9..15 unused */
    double *out_im  = aa(N * K_BATCH);
    double *tw_re   = aa(N * K_BATCH);    /* unused */
    double *tw_im   = aa(N * K_BATCH);    /* unused */

    /* Fill batched: each batch b has 16 random reals. */
    srand(42);
    for (int b = 0; b < K_BATCH; b++) {
        for (int n = 0; n < N; n++) {
            in_re[(size_t)n * K_BATCH + b] = (double)rand() / RAND_MAX - 0.5;
        }
    }
    memset(in_im, 0, N * K_BATCH * sizeof(double));
    memset(out_re, 0, N * K_BATCH * sizeof(double));
    memset(out_im, 0, N * K_BATCH * sizeof(double));

    radix16_n1_fwd_avx512_gen(in_re, in_im, out_re, out_im, tw_re, tw_im, K_BATCH);

    /* For each batch, compute reference and compare. */
    double max_err = 0;
    int worst_b = -1, worst_k = -1;
    char worst_what = '?';
    for (int b = 0; b < K_BATCH; b++) {
        double x[N], ref_re[N/2+1], ref_im[N/2+1];
        for (int n = 0; n < N; n++) x[n] = in_re[(size_t)n * K_BATCH + b];
        ref_r2c_fwd(x, ref_re, ref_im);
        for (int k = 0; k <= N/2; k++) {
            double got_re = out_re[(size_t)k * K_BATCH + b];
            double got_im = out_im[(size_t)k * K_BATCH + b];
            double er = fabs(got_re - ref_re[k]);
            double ei = fabs(got_im - ref_im[k]);
            if (er > max_err) { max_err = er; worst_b = b; worst_k = k; worst_what = 'r'; }
            if (ei > max_err) { max_err = ei; worst_b = b; worst_k = k; worst_what = 'i'; }
        }
    }
    printf("R=16 r2c forward verification\n");
    printf("  N=%d, K=%d, batches=%d\n", N, K_BATCH, K_BATCH);
    printf("  max abs error: %.3e  (worst: batch=%d k=%d %c-part)\n",
           max_err, worst_b, worst_k, worst_what);

    /* Show first batch detail */
    printf("\n  Detail batch 0:\n");
    printf("  %-3s %-22s %-22s %-12s\n", "k", "ref (Re, Im)", "got (Re, Im)", "abs_err");
    double x0[N];
    for (int n = 0; n < N; n++) x0[n] = in_re[(size_t)n * K_BATCH + 0];
    double r_re[N/2+1], r_im[N/2+1];
    ref_r2c_fwd(x0, r_re, r_im);
    for (int k = 0; k <= N/2; k++) {
        double g_re = out_re[(size_t)k * K_BATCH + 0];
        double g_im = out_im[(size_t)k * K_BATCH + 0];
        double er = fmax(fabs(g_re - r_re[k]), fabs(g_im - r_im[k]));
        printf("  %-3d (%+.6f, %+.6f) (%+.6f, %+.6f) %.2e\n",
               k, r_re[k], r_im[k], g_re, g_im, er);
    }

    return max_err < 1e-10 ? 0 : 1;
}
